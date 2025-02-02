// This file has been migrated to TypeScript.
// See: cgm/viz/static/js/components/graph-visualization.ts

class GraphVisualization {
    setupResizers() {
        this.setupVerticalResizer();
        this.setupHorizontalResizer();
    }

    setupVerticalResizer() {
        const resizer = document.getElementById('vertical-resizer');
        const graphContainer = document.querySelector('.graph-container');
        const panelsContainer = document.querySelector('.panels-container');
        
        let isResizing = false;
        let startX;
        let startGraphWidth;
        let startPanelsWidth;
        
        const startResize = (e) => {
            isResizing = true;
            resizer.classList.add('resizing');
            startX = e.pageX;
            startGraphWidth = graphContainer.offsetWidth;
            startPanelsWidth = panelsContainer.offsetWidth;
            document.documentElement.style.cursor = 'col-resize';
            e.preventDefault();
            e.stopPropagation();
        };
        
        const resize = (e) => {
            if (!isResizing) return;
            
            const dx = e.pageX - startX;
            
            // Calculate new widths
            const newGraphWidth = startGraphWidth + dx;
            const newPanelsWidth = startPanelsWidth - dx;
            
            // Apply minimum widths
            if (newGraphWidth >= 200 && newPanelsWidth >= 200) {
                graphContainer.style.flex = 'none';
                panelsContainer.style.flex = 'none';
                graphContainer.style.width = `${newGraphWidth}px`;
                panelsContainer.style.width = `${newPanelsWidth}px`;
                
                // Update visualization width
                this.width = this.calculateWidth();
                this.svg.attr("width", this.width);
                
                // Update force simulation center
                this.simulation.force("x", d3.forceX(this.width / 2).strength(0.05));
                this.simulation.alpha(0.3).restart();
            }
            e.preventDefault();
            e.stopPropagation();
        };
        
        const stopResize = (e) => {
            if (!isResizing) return;
            isResizing = false;
            resizer.classList.remove('resizing');
            document.documentElement.style.cursor = '';
            e.preventDefault();
            e.stopPropagation();
        };
        
        resizer.addEventListener('mousedown', startResize);
        document.addEventListener('mousemove', resize);
        document.addEventListener('mouseup', stopResize);

        // Add window resize handler
        window.addEventListener('resize', () => {
            // Reset flex layout
            graphContainer.style.flex = '1 1 70%';
            panelsContainer.style.flex = '1 1 30%';
            graphContainer.style.width = '';
            panelsContainer.style.width = '';
            
            // Update visualization dimensions
            this.width = this.calculateWidth();
            this.height = window.innerHeight;
            this.svg.attr("width", this.width)
                .attr("height", this.height);
            this.simulation.force("x", d3.forceX(this.width / 2).strength(0.05))
                .force("y", d3.forceY(this.height / 2).strength(0.05));
            this.simulation.alpha(0.3).restart();
        });
    }

    setupHorizontalResizer() {
        const resizer = document.getElementById('horizontal-resizer');
        const upperPanel = document.querySelector('.upper-panel');
        const lowerPanel = document.querySelector('.lower-panel');
        
        let isResizing = false;
        let startY;
        let startUpperHeight;
        let startLowerHeight;
        
        const startResize = (e) => {
            isResizing = true;
            resizer.classList.add('resizing');
            startY = e.pageY;
            startUpperHeight = upperPanel.offsetHeight;
            startLowerHeight = lowerPanel.offsetHeight;
            document.documentElement.style.cursor = 'row-resize';
            e.preventDefault();
            e.stopPropagation();
        };
        
        const resize = (e) => {
            if (!isResizing) return;
            
            const dy = e.pageY - startY;
            
            // Calculate new heights
            const newUpperHeight = startUpperHeight + dy;
            const newLowerHeight = startLowerHeight - dy;
            
            // Apply minimum heights
            if (newUpperHeight >= 100 && newLowerHeight >= 100) {
                upperPanel.style.flex = 'none';
                lowerPanel.style.flex = 'none';
                upperPanel.style.height = `${newUpperHeight}px`;
                lowerPanel.style.height = `${newLowerHeight}px`;
            }
            e.preventDefault();
            e.stopPropagation();
        };
        
        const stopResize = (e) => {
            if (!isResizing) return;
            isResizing = false;
            resizer.classList.remove('resizing');
            document.documentElement.style.cursor = '';
            e.preventDefault();
            e.stopPropagation();
        };
        
        resizer.addEventListener('mousedown', startResize);
        document.addEventListener('mousemove', resize);
        document.addEventListener('mouseup', stopResize);

        // Add window resize handler for vertical layout
        window.addEventListener('resize', () => {
            // Reset flex layout
            upperPanel.style.flex = '1 1 50%';
            lowerPanel.style.flex = '1 1 50%';
            upperPanel.style.height = '';
            lowerPanel.style.height = '';
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
        
        // Set up resizers
        this.setupResizers();
        
        // Initialize plot manager
        this.plotManager = new PlotManager();
        
        // Start update loop
        this.startUpdateLoop();
        
        this.conditioned_vars = {};
    }

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

    updatePanel(node) {
        // This method has been moved into handleNodeClick
    }

    updateTableHighlighting(node) {
        const table = d3.select(".upper-panel .cpd-table");
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
                        ...node,  // This includes conditioned_state from server
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
                    // For non-dragged nodes, preserve position but update state
                    const newNodeState = {
                        ...node,  // This includes conditioned_state from server
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
                    ...node,  // This includes conditioned_state from server
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

        // Update plot if we have a selected node
        if (this.selectedNode) {
            const updatedNode = newData.nodes.find(n => n.id === this.selectedNode.id);
            if (updatedNode) {
                // Update the selected node reference
                this.selectedNode = updatedNode;
                
                // Update table highlighting
                this.updateTableHighlighting(updatedNode);
                
                // Update plot data if it exists
                if (this.plotManager.plots.has('selected-node')) {
                    const plotData = {
                        variable: updatedNode.id,
                        title: `Distribution for ${updatedNode.id}`,
                        samples: this.generateDummySamples(updatedNode)
                    };
                    this.plotManager.updatePlot('selected-node', plotData);
                }
            }
        }
    }
}
// Keeping this file as a placeholder for any remaining unmigrated code. 