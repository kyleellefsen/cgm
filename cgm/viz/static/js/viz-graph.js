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
        console.log("=== GRAPH VISUALIZATION INITIALIZING ===");
        
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

        // Create force simulation
        this.simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-500))
            .force("collide", d3.forceCollide().radius(50))
            .force("x", d3.forceX(this.width / 2).strength(0.05))
            .force("y", d3.forceY(this.height / 2).strength(0.05))
            .on("tick", () => this.tick());
            
        // Add containers for different elements
        this.linksGroup = this.svg.append("g");
        this.nodesGroup = this.svg.append("g");
        this.labelsGroup = this.svg.append("g");
        
        // Set up resizer
        this.setupResizer();
        
        console.log("=== GRAPH VISUALIZATION INITIALIZED ===");
        
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
        event.sourceEvent.stopPropagation();  // Prevent drag from interfering with updates
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
        event.sourceEvent.stopPropagation();  // Prevent drag from interfering with updates
    }

    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        // Keep nodes fixed where they are dropped
        event.sourceEvent.stopPropagation();  // Prevent drag from interfering with updates
    }

    // Handle node selection and conditioning
    handleNodeClick(event, d) {
        event.stopPropagation();
        
        console.log(`Node ${d.id} clicked:`, {
            currentState: d.conditioned_state,
            willTransitionTo: d.conditioned_state === 0 ? -1 : 0
        });
        
        // Remove selection class from all nodes
        this.nodesGroup.selectAll(".node")
            .classed("selected", false);
        
        // Add selection class to clicked node
        d3.select(event.currentTarget).classed("selected", true);
        this.selectedNode = d;
        
        // Update panel
        this.updatePanel(d);
        
        // Toggle node conditioning
        const newState = d.conditioned_state === 0 ? -1 : 0;
        console.log(`Sending condition request for node ${d.id}:`, {
            newState,
            url: `/condition/${d.id}/${newState}`
        });
        
        fetch(`/condition/${d.id}/${newState}`, {
            method: 'POST'
        }).then(() => {
            console.log(`Condition request completed for node ${d.id}, fetching new state`);
            // Refresh graph state after conditioning
            this.fetchAndUpdateState();
        });
    }
    
    // Update panel with CPD table
    updatePanel(node) {
        const panel = d3.select(".panel-container");
        
        if (!node.cpd) {
            panel.html(`
                <div class="panel-header">Node ${node.id}</div>
                <div class="panel-placeholder">No CPD available for this node</div>
            `);
            return;
        }
        
        panel.html(`
            <div class="panel-header">CPD for Node ${node.id}</div>
            <div class="cpd-content">${node.cpd}</div>
        `);
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
        console.log("=== UPDATE DATA START ===");
        console.log("Raw data from server:", JSON.stringify(newData, null, 2));
        
        // Update existing nodes and add new ones
        newData.nodes.forEach(node => {
            const existingNode = this.simulationNodes.get(node.id);
            const nodeWidth = this.calculateNodeWidth(node.id);
            
            console.log(`Processing node ${node.id}:`, {
                isExisting: !!existingNode,
                incomingState: node.conditioned_state,
                currentState: existingNode?.conditioned_state
            });
            
            if (existingNode) {
                // Update existing node while preserving position and state
                const beforeUpdate = {...existingNode};
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
                console.log(`Updated existing node ${node.id}:`, {
                    before: {
                        conditioned_state: beforeUpdate.conditioned_state,
                        x: beforeUpdate.x,
                        y: beforeUpdate.y
                    },
                    after: {
                        conditioned_state: existingNode.conditioned_state,
                        x: existingNode.x,
                        y: existingNode.y
                    }
                });
            } else {
                // Add new node
                const newNode = {
                    ...node,
                    x: this.width/2 + (Math.random() - 0.5) * 100,
                    y: this.height/2 + (Math.random() - 0.5) * 100,
                    fx: null,
                    fy: null,
                    vx: 0,
                    vy: 0,
                    type: node.type || 'effect',
                    width: nodeWidth,
                    height: this.nodeHeight
                };
                this.simulationNodes.set(node.id, newNode);
                console.log(`Added new node ${node.id}:`, {
                    conditioned_state: newNode.conditioned_state,
                    position: {x: newNode.x, y: newNode.y}
                });
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
            .force("charge", d3.forceManyBody().strength(-500))
            .force("collide", d3.forceCollide().radius(50))
            .force("x", d3.forceX(this.width / 2).strength(0.05))
            .force("y", d3.forceY(this.height / 2).strength(0.05))
            .alpha(0.3)
            .restart();
        
        // Update visual elements
        this.updateVisuals();
        
        console.log("=== UPDATE DATA END ===");
    }
    
    // Update visual elements without touching simulation
    updateVisuals() {
        console.log("=== UPDATE VISUALS START ===");
        
        // Get current nodes with their data
        const currentNodes = Array.from(this.simulationNodes.values());
        console.log("Current nodes before visual update:", currentNodes.map(d => ({
            id: d.id,
            conditioned_state: d.conditioned_state,
            isConditioned: d.conditioned_state >= 0
        })));
            
        // Update nodes
        const nodes = this.nodesGroup
            .selectAll("g.node")
            .data(currentNodes, d => d.id);
            
        // Remove old nodes
        nodes.exit().remove();
        
        // Enter new nodes
        const nodeEnter = nodes.enter()
            .append("g")
            .attr("class", d => `node ${d.type || 'effect'}`);
            
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
            
        // Add conditioned circle on top
        nodeEnter.append("circle")
            .attr("class", "conditioned")
            .attr("r", d => Math.min(d.width, d.height) * 0.6)
            .style("stroke", "#2563eb")
            .style("stroke-width", "4px")
            .style("fill", "none")
            .style("opacity", d => {
                const isConditioned = d.conditioned_state >= 0;
                console.log(`Setting initial circle opacity for node ${d.id}:`, {
                    conditioned_state: d.conditioned_state,
                    isConditioned,
                    opacity: isConditioned ? 1 : 0,
                    element: 'circle.conditioned'
                });
                return isConditioned ? 1 : 0;
            })
            .style("display", "block");
            
        // Add text label on top
        nodeEnter.append("text")
            .attr("class", "node-label")
            .text(d => d.id);
            
        // Update all nodes (both new and existing)
        const allNodes = nodes.merge(nodeEnter);
        
        // Update node elements
        allNodes.each(function(d) {
            const node = d3.select(this);
            
            // Update ellipse
            node.select("ellipse")
                .attr("rx", d.width / 2)
                .attr("ry", d.height / 2);
                
            // Update conditioned circle
            const circle = node.select("circle.conditioned");
            const isConditioned = d.conditioned_state >= 0;
            
            console.log(`Updating circle for node ${d.id}:`, {
                conditioned_state: d.conditioned_state,
                isConditioned,
                newOpacity: isConditioned ? 1 : 0,
                currentStyle: {
                    opacity: circle.style("opacity"),
                    display: circle.style("display"),
                    stroke: circle.style("stroke"),
                    'stroke-width': circle.style("stroke-width")
                }
            });
            
            circle
                .attr("r", Math.min(d.width, d.height) * 0.6)
                .style("opacity", isConditioned ? 1 : 0)
                .style("stroke", "#2563eb")
                .style("stroke-width", "4px")
                .style("fill", "none");
                
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
            
        console.log("=== UPDATE VISUALS END ===");
    }
    
    async startUpdateLoop() {
        while (true) {
            try {
                console.log("=== FETCH START ===");
                const response = await fetch('/state');
                const data = await response.json();
                console.log("Raw response from /state:", JSON.stringify(data, null, 2));
                console.log("=== FETCH END ===");
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
console.log("=== SCRIPT LOADED ===");
window.addEventListener('DOMContentLoaded', () => {
    console.log("=== DOM LOADED ===");
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