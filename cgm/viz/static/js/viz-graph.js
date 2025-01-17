class GraphVisualization {
    constructor() {
        // Set up constants
        this.width = window.innerWidth * 0.7;  // Adjust for panel
        this.height = window.innerHeight;
        
        // Initialize simulation state
        this.simulationNodes = new Map(); // Store simulation nodes by ID
        this.simulationLinks = new Map(); // Store simulation links by ID
        this.selectedNode = null;
        
        // Set up SVG
        // Select the SVG element inside the graph container
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
            .force("link", d3.forceLink().id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-800))
            .force("x", d3.forceX(this.width / 2))
            .force("y", d3.forceY(this.height / 2))
            .on("tick", () => this.tick());
            
        // Add containers
        this.linksGroup = this.svg.append("g");
        this.nodesGroup = this.svg.append("g");
        this.labelsGroup = this.svg.append("g");
        
        // Start update loop
        this.startUpdateLoop();
    }
    
    // Drag handlers
    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.1);
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    // Handle node selection
    handleNodeClick(event, d) {
        // Remove previous selection
        this.nodesGroup.selectAll(".node").classed("selected", false);
        
        // Update selection
        const node = d3.select(event.currentTarget);
        node.classed("selected", true);
        this.selectedNode = d;
        
        // Update panel
        this.updatePanel(d);
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
        this.linksGroup.selectAll(".link")
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        this.nodesGroup.selectAll(".node")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        this.labelsGroup.selectAll(".node-label")
            .attr("x", d => d.x)
            .attr("y", d => d.y);

        this.labelsGroup.selectAll(".node-states")
            .attr("x", d => d.x)
            .attr("y", d => d.y + 25);
    }
    
    // Update data without disturbing simulation
    updateData(newData) {
        // Check if graph structure has actually changed
        const structureChanged = this.hasStructureChanged(newData);
        
        if (structureChanged) {
            // Update simulation nodes while preserving positions
            newData.nodes.forEach(node => {
                if (!this.simulationNodes.has(node.id)) {
                    // New node: add to simulation
                    const simNode = {...node, x: this.width/2, y: this.height/2};
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
    
    // Check if graph structure changed
    hasStructureChanged(newData) {
        // Check if number of nodes changed
        if (this.simulationNodes.size !== newData.nodes.length) return true;
        
        // Check if number of links changed
        if (this.simulationLinks.size !== newData.links.length) return true;
        
        // Check if any nodes changed
        for (const node of newData.nodes) {
            if (!this.simulationNodes.has(node.id)) return true;
        }
        
        // Check if any links changed
        for (const link of newData.links) {
            const linkId = `${link.source}-${link.target}`;
            if (!this.simulationLinks.has(linkId)) return true;
        }
        
        return false;
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
        
        nodes.enter()
            .append("circle")
            .attr("class", "node")
            .attr("r", 15)
            .call(drag)
            .on("click", (e,d) => this.handleNodeClick(e,d))
            .merge(nodes);
            
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
            
        // Update labels
        const labels = this.labelsGroup
            .selectAll(".node-label")
            .data(Array.from(this.simulationNodes.values()), d => d.id);
            
        labels.exit().remove();
        
        labels.enter()
            .append("text")
            .attr("class", "node-label")
            .merge(labels)
            .text(d => d.id);
            
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
