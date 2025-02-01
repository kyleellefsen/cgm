class GraphVisualization {
    setupResizer() {
        const resizer = document.getElementById('resizer');
        const graphContainer = document.querySelector('.graph-container');
        const panelContainer = document.querySelector('.panel-container');
        
        let isResizing = false;
        let startX;
        let startGraphWidth;
        let startPanelWidth;
        
        // Handle resize start
        const startResize = (e) => {
            isResizing = true;
            resizer.classList.add('resizing');
            startX = e.pageX;
            startGraphWidth = graphContainer.offsetWidth;
            startPanelWidth = panelContainer.offsetWidth;
            document.documentElement.style.cursor = 'col-resize';
        };
        
        // Handle resize
        const resize = (e) => {
            if (!isResizing) return;
            
            const dx = e.pageX - startX;
            
            // Calculate new widths
            const newGraphWidth = startGraphWidth + dx;
            const newPanelWidth = startPanelWidth - dx;
            
            // Apply minimum widths
            if (newGraphWidth >= 200 && newPanelWidth >= 200) {
                graphContainer.style.flex = 'none';
                panelContainer.style.flex = 'none';
                graphContainer.style.width = `${newGraphWidth}px`;
                panelContainer.style.width = `${newPanelWidth}px`;
                
                // Update visualization width
                this.width = this.calculateWidth();
                this.svg.attr("width", this.width);
                
                // Update force simulation center
                this.simulation.force("x", d3.forceX(this.width / 2).strength(0.05));
                this.simulation.alpha(0.3).restart();
            }
        };
        
        // Handle resize end
        const stopResize = () => {
            if (!isResizing) return;
            isResizing = false;
            resizer.classList.remove('resizing');
            document.documentElement.style.cursor = '';
        };
        
        // Add event listeners
        resizer.addEventListener('mousedown', startResize);
        document.addEventListener('mousemove', resize);
        document.addEventListener('mouseup', stopResize);
        
        // Clean up on window resize
        window.addEventListener('resize', () => {
            this.width = this.calculateWidth();
            this.height = window.innerHeight;
            this.svg.attr("width", this.width)
                .attr("height", this.height);
            this.simulation.force("x", d3.forceX(this.width / 2).strength(0.05))
                .force("y", d3.forceY(this.height / 2).strength(0.05));
            this.simulation.alpha(0.3).restart();
        });
    }
    
    calculateWidth() {
        return document.querySelector('.graph-container').offsetWidth;
    }

    constructor() {
        
        // Set up constants
        this.width = this.calculateWidth();
        this.height = window.innerHeight;
        this.nodeHeight = 30;  // Fixed height for nodes
        this.textPadding = 20; // Padding on each side of text
        
        // Initialize simulation state
        this.simulationNodes = new Map(); // Store simulation nodes by ID
        this.simulationLinks = new Map(); // Store simulation links by ID
        this.selectedNode = null;
        
        // Set up SVG - clear existing content first
        d3.select(".graph-container").selectAll("svg").remove();
        this.svg = d3.select(".graph-container").append("svg")
            .attr("width", this.width)
            .attr("height", this.height);
            
        // Add arrow marker
        this.svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 8)
            .attr("markerHeight", 8)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");

        // Initialize simulation with gentler forces
        this.simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("collide", d3.forceCollide().radius(50))
            .force("x", d3.forceX(this.width / 2).strength(0.03))
            .force("y", d3.forceY(this.height / 2).strength(0.03))
            .alphaDecay(0.02)  // Slower decay for smoother motion
            .velocityDecay(0.3)  // More damping for stability
            .on("tick", () => this.tick());
            
        // Add containers for different elements
        this.linksGroup = this.svg.append("g");
        this.nodesGroup = this.svg.append("g");
        this.labelsGroup = this.svg.append("g");
        
        // Set up resizer
        this.setupResizer();
        
        
        // Start update loop
        this.startUpdateLoop();
        
        this.conditioned_vars = {};
    }
    
    // Calculate node width based on text
    calculateNodeWidth(text) {
        // Create temporary text element to measure width
        const temp = this.svg.append("text")
            .attr("class", "node-label")
            .text(text);
        const width = temp.node().getBBox().width;
        temp.remove();
        return width + this.textPadding * 2; // Add padding on both sides
    }
    
    // Drag handlers
    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        d.isDragging = true;
        event.sourceEvent.stopPropagation();
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
        event.sourceEvent.stopPropagation();
    }

    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        // Only keep position fixed if the node was explicitly pinned
        if (!d.isPinned) {
            d.fx = null;
            d.fy = null;
        }
        d.isDragging = false;
        event.sourceEvent.stopPropagation();
    }

    // Handle node selection and conditioning
    handleNodeClick(e, d) {
        e.stopPropagation();
        this.updatePanel(d);
        e.sourceEvent.stopPropagation();
    }
    
    // Update panel with CPD table
    updatePanel(node) {
        const panel = d3.select(".panel-container");
        
        if (!node.cpd) {
            panel.html(`
                <div class="panel-header">Node ${node.id}</div>
                <div class="panel-placeholder">No CPD available</div>
            `);
            return;
        }

        // Store the selected node for later updates
        this.selectedNode = node;

        // Use the actual CPD HTML from the server
        panel.html(`
            <div class="panel-header">CPD for ${node.id}</div>
            <div class="cpd-content">
                ${node.cpd}
            </div>
        `);

        // Determine if this node has parents by checking for multiple rows
        const table = panel.select(".cpd-table");
        const hasParents = table.selectAll("tbody tr").size() > 1;
        table.classed("no-parents", !hasParents);

        // Add click handlers for state cells
        panel.selectAll("[data-variable][data-value]")
            .on("click", (event) => {
                const cell = event.target;
                const variable = cell.dataset.variable;
                const value = parseInt(cell.dataset.value);
                const currentState = parseInt(node.conditioned_state);
                
                // Only allow conditioning on the clicked node
                if (variable === node.id) {
                    const newState = currentState === value ? -1 : value;
                    
                    fetch(`/condition/${node.id}/${newState}`, { method: 'POST' })
                        .then(() => this.fetchAndUpdateState())
                        .catch(error => console.error('Condition failed:', error));
                }
            });

        // Apply initial highlighting if node is already conditioned
        this.updateTableHighlighting(node);
    }

    // Update table highlighting based on node's conditioning state
    updateTableHighlighting(node) {
        const table = d3.select(".cpd-table");
        if (!table.node()) return;  // Exit if no table is displayed

        // Clear any existing highlighting
        table.selectAll("td, th").classed("state-active", false);

        const currentState = parseInt(node.conditioned_state);
        if (currentState === -1) return;  // No highlighting needed

        const hasParents = !table.classed("no-parents");
        
        if (hasParents) {
            // For nodes with parents, highlight cells and header showing the current state
            table.selectAll(`td[data-variable="${node.id}"][data-value="${currentState}"], 
                           th[data-variable="${node.id}"][data-value="${currentState}"]`)
                .classed("state-active", true);
        } else {
            // For nodes without parents, highlight the column corresponding to the state
            // Add 1 to account for the label column
            const stateColumn = currentState + 1;
            table.selectAll(`td:nth-child(${stateColumn}), 
                           th:nth-child(${stateColumn})`)
                .classed("state-active", true);
        }
    }
    
    generateCPDRows(node) {
        // Simple example - adapt based on your actual CPD structure
        return `
            <tr>
                <th>State</th>
                <th>Probability</th>
            </tr>
            ${Array.from({length: node.states}, (_, i) => `
                <tr>
                    <td class="state-cell" 
                        data-variable="${node.id}" 
                        data-state="${i}"
                        style="cursor: pointer;
                               background: ${node.conditioned_state === i ? '#e0f2fe' : ''}">
                        State ${i}
                    </td>
                    <td>${(Math.random().toFixed(2))}</td>
                </tr>
            `).join('')}
        `;
    }
    
    // Update the visual elements based on simulation state
    tick() {
        // Update links
        this.linksGroup.selectAll(".link")
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        // Update node groups
        this.nodesGroup.selectAll(".node")
            .attr("transform", d => `translate(${d.x},${d.y})`);

        // Update labels
        this.labelsGroup.selectAll(".node-states")
            .attr("x", d => d.x)
            .attr("y", d => d.y + this.nodeHeight);
    }
    
    // Check if graph structure changed
    hasStructureChanged(newData) {
        if (this.simulationNodes.size !== newData.nodes.length) return true;
        if (this.simulationLinks.size !== newData.links.length) return true;
        
        for (const node of newData.nodes) {
            if (!this.simulationNodes.has(node.id)) return true;
        }
        
        for (const link of newData.links) {
            const linkId = `${link.source}-${link.target}`;
            if (!this.simulationLinks.has(linkId)) return true;
        }
        
        return false;
    }
    
    // Update data without disturbing simulation
    updateData(newData) {
        // Update existing nodes and add new ones
        newData.nodes.forEach(node => {
            const existingNode = this.simulationNodes.get(node.id);
            const nodeWidth = this.calculateNodeWidth(node.id);
            
            if (existingNode) {
                // Update existing node while preserving position and state
                if (existingNode.isDragging) {
                    // Preserve all motion state during drag
                    Object.assign(existingNode, {
                        ...node,
                        width: nodeWidth,
                        height: this.nodeHeight,
                        x: existingNode.x,
                        y: existingNode.y,
                        fx: existingNode.fx,
                        fy: existingNode.fy,
                        vx: existingNode.vx,
                        vy: existingNode.vy
                    });
                } else {
                    // For non-dragged nodes, only preserve pinned state
                    const newNodeState = {
                        ...node,
                        width: nodeWidth,
                        height: this.nodeHeight
                    };
                    
                    if (existingNode.isPinned) {
                        newNodeState.fx = existingNode.x;
                        newNodeState.fy = existingNode.y;
                    } else {
                        // Preserve some momentum for smoother updates
                        newNodeState.x = existingNode.x;
                        newNodeState.y = existingNode.y;
                        newNodeState.vx = existingNode.vx * 0.5;
                        newNodeState.vy = existingNode.vy * 0.5;
                    }
                    
                    Object.assign(existingNode, newNodeState);
                }
            } else {
                // Add new node with initial position
                const newNode = {
                    ...node,
                    x: this.width/2 + (Math.random() - 0.5) * 100,
                    y: this.height/2 + (Math.random() - 0.5) * 100,
                    fx: null,
                    fy: null,
                    vx: 0,
                    vy: 0,
                    type: node.type || 'node',
                    width: nodeWidth,
                    height: this.nodeHeight
                };
                this.simulationNodes.set(node.id, newNode);
            }
        });
        
        // Remove nodes that no longer exist
        for (const [id, node] of this.simulationNodes.entries()) {
            if (!newData.nodes.find(n => n.id === id)) {
                this.simulationNodes.delete(id);
            }
        }
        
        // Update links
        this.simulationLinks.clear();
        newData.links.forEach(link => {
            const sourceNode = this.simulationNodes.get(link.source);
            const targetNode = this.simulationNodes.get(link.target);
            if (sourceNode && targetNode) {
                const linkId = `${link.source}-${link.target}`;
                this.simulationLinks.set(linkId, {
                    ...link,
                    source: sourceNode,
                    target: targetNode
                });
            }
        });
        
        // Update simulation with current data
        const nodes = Array.from(this.simulationNodes.values());
        const links = Array.from(this.simulationLinks.values());
        
        this.simulation
            .nodes(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("collide", d3.forceCollide().radius(50))
            .force("x", d3.forceX(this.width / 2).strength(0.03))
            .force("y", d3.forceY(this.height / 2).strength(0.03))
            .alpha(0.1)  // Gentler simulation restart
            .restart();
        
        // Update visual elements
        this.updateVisuals();

        // Update any open CPD table
        if (this.selectedNode) {
            const updatedNode = newData.nodes.find(n => n.id === this.selectedNode.id);
            if (updatedNode) {
                this.updateTableHighlighting(updatedNode);
            }
        }
    }
    
    // Update visual elements without touching simulation
    updateVisuals() {
        
        // Get current nodes with their data
        const currentNodes = Array.from(this.simulationNodes.values());
            
            
        // Update nodes
        const nodes = this.nodesGroup
            .selectAll("g.node")
            .data(currentNodes, d => d.id);
            
        // Remove old nodes
        nodes.exit().remove();
        
        // Enter new nodes
        const nodeEnter = nodes.enter()
            .append("g")
            .attr("class", d => `node ${d.type || 'node'}`);
            
        // Add basic elements to new nodes
        nodeEnter
            .call(d3.drag()
                .on("start", (e,d) => this.dragstarted(e,d))
                .on("drag", (e,d) => this.dragged(e,d))
                .on("end", (e,d) => this.dragended(e,d)))
            .on("click", (e,d) => this.handleNodeClick(e,d));
            
        // Add ellipse first (background)
        nodeEnter.append("ellipse")
            .attr("rx", d => d.width / 2)
            .attr("ry", d => d.height / 2);
            
        // Add text label on top
        nodeEnter.append("text")
            .attr("class", "node-label")
            .text(d => d.id);
            
        // Update all nodes (both new and existing)
        const allNodes = nodes.merge(nodeEnter);
        
        // Update node elements
        allNodes.each(function(d) {
            const node = d3.select(this);
            const isConditioned = d.conditioned_state >= 0;
            
            // Toggle 'conditioned' class
            node.classed("conditioned", isConditioned);
            
            // Update ellipse styling
            node.select("ellipse")
                .attr("rx", d.width / 2)
                .attr("ry", d.height / 2);
                
            // Update text
            node.select("text.node-label")
                .text(d.id);
        });
        
        // Update links
        const links = this.linksGroup
            .selectAll("line.link")
            .data(Array.from(this.simulationLinks.values()), 
                d => `${d.source.id}-${d.target.id}`);
            
        links.exit().remove();
        
        const linkEnter = links.enter()
            .append("line")
            .attr("class", "link");
            
        // Update all links
        links.merge(linkEnter);
        
        // Update state labels
        const states = this.labelsGroup
            .selectAll("text.node-states")
            .data(currentNodes, d => d.id);
            
        states.exit().remove();
        
        const stateEnter = states.enter()
            .append("text")
            .attr("class", "node-states");
            
        states.merge(stateEnter)
            .text(d => `states: ${d.states}`);
            
    }
    
    async startUpdateLoop() {
        while (true) {
            try {
                const response = await fetch('/state');
                const data = await response.json();
                this.updateData(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    
    fetchAndUpdateState() {
        fetch('/state')
            .then(response => response.json())
            .then(data => this.updateData(data));
    }

    highlightConditionedRows() {
        const table = document.querySelector('.cpd-table');
        if (!table) return;
        
        const conditionedVars = JSON.parse(table.dataset.conditioned || "{}");
        
        table.querySelectorAll('tr').forEach(row => {
            const matches = Array.from(row.cells).every(cell => {
                const varName = cell.getAttribute('data-variable');
                const value = parseInt(cell.getAttribute('data-value'));
                return !varName || !(varName in conditionedVars) || 
                       conditionedVars[varName] === value;
            });
            row.classList.toggle('active', matches);
        });
    }
}

// Create instance when the script loads
window.addEventListener('DOMContentLoaded', () => {
    try {
        window.graphViz = new GraphVisualization();
    } catch (error) {
        console.error("=== ERROR INITIALIZING GRAPH VISUALIZATION ===");
        console.error(error);
    }
});

// Add error handler for uncaught errors
window.addEventListener('error', (event) => {
    console.error("=== UNCAUGHT ERROR ===");
    console.error(event.error);
});

// Add error handler for unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error("=== UNHANDLED PROMISE REJECTION ===");
    console.error(event.reason);
});