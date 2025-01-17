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
        // Set up resizer
        this.setupResizer();
        
        // Set up constants
        this.width = this.calculateWidth();
        this.height = window.innerHeight;
        this.nodeHeight = 30;  // Fixed height for nodes
        this.textPadding = 20; // Padding on each side of text
        
        // Initialize simulation state
        this.simulationNodes = new Map(); // Store simulation nodes by ID
        this.simulationLinks = new Map(); // Store simulation links by ID
        this.selectedNode = null;
        
        // Set up SVG
        this.svg = d3.select(".graph-container").select("svg")
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
        
        // Start update loop
        this.startUpdateLoop();
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
        if (!event.active) this.simulation.alphaTarget(0.3).restart();  // Increased alpha target
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
        // Heat up the simulation during drag
        this.simulation.alpha(0.3).restart();
    }

    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        // Optional: keep the node fixed where it was dropped
        // Comment these out if you want nodes to keep their positions after drag
        d.fx = null;
        d.fy = null;
    }

    // Handle node selection and conditioning
    handleNodeClick(event, d) {
        // Remove selection class from all nodes
        this.nodesGroup.selectAll(".node")
            .classed("selected", false);
        
        // Add selection class to clicked node
        const node = d3.select(event.currentTarget);
        node.classed("selected", true);
        
        this.selectedNode = d;
        
        // Update panel
        this.updatePanel(d);
        
        // Toggle conditioned state if applicable
        if (d.canBeConditioned) {
            const ellipse = node.select("ellipse");
            const isConditioned = ellipse.classed("conditioned");
            ellipse.classed("conditioned", !isConditioned);
            d.isConditioned = !isConditioned;
        }
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
        const structureChanged = this.hasStructureChanged(newData);
        
        if (structureChanged) {
            // Update simulation nodes while preserving positions
            newData.nodes.forEach(node => {
                if (!this.simulationNodes.has(node.id)) {
                    // Calculate node width based on text
                    const nodeWidth = this.calculateNodeWidth(node.id);
                    
                    // New node: add to simulation
                    const simNode = {
                        ...node,
                        x: this.width/2,
                        y: this.height/2,
                        type: node.type || 'effect',  // Default to effect if no type specified
                        width: nodeWidth,
                        height: this.nodeHeight
                    };
                    this.simulationNodes.set(node.id, simNode);
                }
            });
            
            // Remove old nodes
            for (const [id, node] of this.simulationNodes.entries()) {
                if (!newData.nodes.find(n => n.id === id)) {
                    this.simulationNodes.delete(id);
                }
            }
            
            // Update simulation links
            this.simulationLinks = new Map(
                newData.links.map(link => [
                    `${link.source}-${link.target}`,
                    {
                        ...link,
                        source: this.simulationNodes.get(link.source),
                        target: this.simulationNodes.get(link.target)
                    }
                ])
            );
            
            // Update simulation with preserved nodes
            this.simulation
                .nodes(Array.from(this.simulationNodes.values()))
                .force("link").links(Array.from(this.simulationLinks.values()));
                
            // Gentle restart
            this.simulation.alpha(0.1).restart();
        }
        
        // Update visual elements
        this.updateVisuals(newData);
    }
    
    // Update visual elements without touching simulation
    updateVisuals(data) {
        const drag = d3.drag()
            .on("start", (e,d) => this.dragstarted(e,d))
            .on("drag", (e,d) => this.dragged(e,d))
            .on("end", (e,d) => this.dragended(e,d));
            
        // Update nodes
        const nodes = this.nodesGroup
            .selectAll(".node")
            .data(Array.from(this.simulationNodes.values()), d => d.id);
            
        nodes.exit().remove();
        
        // Enter new nodes
        const nodeEnter = nodes.enter()
            .append("g")
            .attr("class", d => `node ${d.type || 'effect'}`)
            .call(drag)
            .on("click", (e,d) => this.handleNodeClick(e,d));

        // Add ellipse to each node
        nodeEnter.append("ellipse")
            .attr("rx", d => d.width / 2)
            .attr("ry", d => d.height / 2);
            
        // Add text label within the node
        nodeEnter.append("text")
            .attr("class", "node-label")
            .text(d => d.id);
            
        // Merge and update existing nodes
        const nodeUpdate = nodeEnter.merge(nodes);
        nodeUpdate.attr("class", d => {
            const baseClass = `node ${d.type || 'effect'}`;
            return this.selectedNode && d.id === this.selectedNode.id
                ? `${baseClass} selected`
                : baseClass;
        });
        
        // Update existing ellipses
        nodeUpdate.select("ellipse")
            .attr("rx", d => d.width / 2)
            .attr("ry", d => d.height / 2);
            
        // Update existing labels
        nodeUpdate.select("text")
            .text(d => d.id);
            
        // Update links
        const links = this.linksGroup
            .selectAll(".link")
            .data(Array.from(this.simulationLinks.values()), 
                d => `${d.source.id}-${d.target.id}`);
            
        links.exit().remove();
        
        links.enter()
            .append("line")
            .attr("class", "link")
            .merge(links);
            
        // Update state labels
        const states = this.labelsGroup
            .selectAll(".node-states")
            .data(Array.from(this.simulationNodes.values()), d => d.id);
            
        states.exit().remove();
        
        states.enter()
            .append("text")
            .attr("class", "node-states")
            .merge(states)
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
}

// Initialize visualization when document is ready
document.addEventListener('DOMContentLoaded', () => {
    const viz = new GraphVisualization();
});