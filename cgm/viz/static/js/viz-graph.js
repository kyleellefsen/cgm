class Plot {
    constructor(container, data) {
        this.container = container;
        this.data = data;
        this.margin = {top: 20, right: 20, bottom: 30, left: 40};
        this.initialize();
    }

    initialize() {
        // Only setup container and basic SVG structure
        this.container.html("");
        this.svg = this.container.append("svg")
            .attr("width", this.width + this.margin.left + this.margin.right)
            .attr("height", this.height + this.margin.top + this.margin.bottom)
            .style("display", "block");
        this.plotGroup = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
    }

    get width() {
        return this.container.node().offsetWidth - this.margin.left - this.margin.right;
    }

    get height() {
        return this.container.node().offsetHeight - this.margin.top - this.margin.bottom;
    }

    update(data) {
        const hasChanged = this.shouldUpdate(data);
        if (hasChanged) {
            this.data = data;
            this.render();
        }
    }

    shouldUpdate(newData) {
        // Subclasses should implement their own update detection
        return true;
    }

    render() {
        // This should be abstract - force subclasses to implement
        throw new Error("render() must be implemented by subclass");
    }
}

class DistributionPlot extends Plot {
    constructor(container, data) {
        super(container, data);
        this.margin = {top: 40, right: 20, bottom: 40, left: 50};
        this.xScale = d3.scaleBand()
            .padding(0.1);
        this.yScale = d3.scaleLinear();
        this.previousData = null;  // Store previous data for comparison
    }

    // Helper method to process data
    processData(samples) {
        if (!samples || !samples.length) return [];
        
        // Count occurrences of each state
        const counts = new Map();
        samples.forEach(s => counts.set(s, (counts.get(s) || 0) + 1));
        
        // Convert to probability
        const probs = Array.from(counts.entries()).map(([state, count]) => ({
            state,
            probability: count / samples.length
        }));
        
        // Sort by state
        return probs.sort((a, b) => a.state - b.state);
    }

    // Helper method to check if data has changed
    hasDataChanged(newData) {
        if (!this.previousData) return true;
        if (!newData || !newData.samples) return true;
        
        const oldProbs = this.processData(this.previousData.samples);
        const newProbs = this.processData(newData.samples);
        
        if (oldProbs.length !== newProbs.length) return true;
        
        return newProbs.some((newProb, i) => {
            const oldProb = oldProbs[i];
            return newProb.state !== oldProb.state || 
                   Math.abs(newProb.probability - oldProb.probability) > 0.001;
        });
    }

    initialize() {
        // Clear any existing content
        this.container.html("");
        
        // Create SVG container with explicit dimensions
        this.svg = this.container.append("svg")
            .attr("width", this.width + this.margin.left + this.margin.right)
            .attr("height", this.height + this.margin.top + this.margin.bottom)
            .style("display", "block")
            .style("overflow", "visible");
            
        // Create plot group with margins
        this.plotGroup = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
            
        // Add axes groups but don't populate them yet
        this.xAxis = this.plotGroup.append("g")
            .attr("class", "axis x-axis")
            .attr("transform", `translate(0,${this.height})`);
            
        this.yAxis = this.plotGroup.append("g")
            .attr("class", "axis y-axis");
            
        // Add static elements that don't need frequent updates
        this.svg.append("text")
            .attr("class", "plot-title")
            .attr("text-anchor", "middle")
            .attr("x", this.margin.left + this.width/2)
            .attr("y", 25)
            .text(this.data.title || "Distribution");

        this.plotGroup.append("text")
            .attr("class", "axis-label")
            .attr("text-anchor", "middle")
            .attr("x", this.width/2)
            .attr("y", this.height + 30)
            .text("State");

        this.plotGroup.append("text")
            .attr("class", "axis-label")
            .attr("text-anchor", "middle")
            .attr("transform", `translate(${-35},${this.height/2})rotate(-90)`)
            .text("Probability");
    }

    render() {
        const {samples} = this.data;
        
        // Check if data has actually changed
        if (!this.hasDataChanged(this.data)) {
            return;  // Skip update if data hasn't changed
        }
        
        // Process the data
        const probs = this.processData(samples);
        if (!probs.length) return;

        // Store current data for future comparison
        this.previousData = {...this.data};

        // Update scales
        this.xScale
            .domain(probs.map(d => d.state))
            .range([0, this.width]);
            
        this.yScale
            .domain([0, 1])
            .range([this.height, 0]);
            
        // Update axes only if domain changed
        const xAxis = d3.axisBottom(this.xScale).tickFormat(d => `State ${d}`);
        const yAxis = d3.axisLeft(this.yScale).ticks(5).tickFormat(d3.format(".0%"));
        
        // Use transition for axes only if they need updating
        this.xAxis.transition()
            .duration(200)
            .call(xAxis);
            
        this.yAxis.transition()
            .duration(200)
            .call(yAxis);
        
        // Update bars with object constancy
        const bars = this.plotGroup.selectAll(".bar")
            .data(probs, d => d.state);  // Use state as key for object constancy
            
        // ENTER new bars
        const barsEnter = bars.enter()
            .append("rect")
            .attr("class", "bar")
            .style("fill", "#4682b4")
            // Start new bars at their final x position but at height 0
            .attr("x", d => this.xScale(d.state))
            .attr("width", this.xScale.bandwidth())
            .attr("y", this.height)
            .attr("height", 0);
            
        // EXIT old bars
        bars.exit()
            .transition()
            .duration(200)
            .attr("y", this.height)
            .attr("height", 0)
            .remove();
            
        // UPDATE + ENTER: merge and transition to new positions
        bars.merge(barsEnter)
            .transition()
            .duration(200)
            .attr("x", d => this.xScale(d.state))
            .attr("width", this.xScale.bandwidth())
            .attr("y", d => this.yScale(d.probability))
            .attr("height", d => this.height - this.yScale(d.probability));
    }
}

class PlotManager {
    constructor() {
        // Create a container for plots that coexists with sampling controls
        const lowerPanel = d3.select(".lower-panel");
        
        // Check if plots container already exists
        let plotsContainer = lowerPanel.select(".plots-container");
        if (plotsContainer.empty()) {
            // Create plots container after sampling controls
            plotsContainer = lowerPanel.append("div")
                .attr("class", "plots-container")
                .style("margin-top", "20px");
        }
        
        this.container = plotsContainer;
        this.plots = new Map();  // Store active plots
        
        // Handle panel resizing
        this.handleResize = this.handleResize.bind(this);
        window.addEventListener('resize', this.handleResize);
    }
    
    handleResize() {
        // Update all plots when panel is resized
        this.plots.forEach(plot => {
            plot.initialize();
            plot.render();
        });
    }
    
    createPlot(id, type, data) {
        // First remove any existing plot with this id
        this.removePlot(id);
        
        const plotContainer = this.container
            .append("div")
            .attr("class", "plot-container")
            .attr("id", `plot-${id}`)
            .style("width", "100%")
            .style("height", "300px")  // Fixed height for plots
            .style("position", "relative")
            .style("margin-bottom", "10px");
            
        let plot;
        switch(type) {
            case 'distribution':
                plot = new DistributionPlot(plotContainer, data);
                break;
            default:
                console.warn(`Unknown plot type: ${type}`);
                return;
        }
        
        this.plots.set(id, plot);
        plot.render();  // Explicitly call render after creation
        return plot;
    }
    
    updatePlot(id, data) {
        const plot = this.plots.get(id);
        if (plot) {
            plot.update(data);
        }
    }
    
    removePlot(id) {
        const plot = this.plots.get(id);
        if (plot) {
            plot.container.remove();
            this.plots.delete(id);
        }
    }
}

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
        
        // Initialize sampling controls
        this.samplingControls = new SamplingControls(
            document.querySelector('.lower-panel'),
            this.handleSamplingRequest.bind(this)
        );
        
        // Track current graph state
        this.currentGraphState = {
            conditions: {},
            lastSamplingResult: null
        };
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
            .attr("class", d => {
                const classes = ['node'];
                if (d.type) classes.push(d.type);
                if (d.conditioned_state >= 0) classes.push('conditioned');
                return classes.join(' ');
            });
            
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

    calculateNodeWidth(text) {
        // Create temporary text element to measure width
        const temp = this.svg.append("text")
            .attr("class", "node-label")
            .text(text);
        const width = temp.node().getBBox().width;
        temp.remove();
        return width + this.textPadding * 2; // Add padding on both sides
    }

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

    async handleNodeClick(event, d) {
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }
        
        console.log('Node clicked:', d.id);
        
        // If clicking the same node, just update the table highlighting
        if (this.selectedNode && this.selectedNode.id === d.id) {
            this.updateTableHighlighting(d);
            return;
        }

        const panel = d3.select(".upper-panel");
        
        if (!d.cpd) {
            panel.html(`
                <div class="panel-header">Node ${d.id}</div>
                <div class="panel-placeholder">No CPD available</div>
            `);
            return;
        }

        // Store the selected node for later updates
        this.selectedNode = d;

        // Use the actual CPD HTML from the server
        panel.html(`
            <div class="panel-header">CPD for ${d.id}</div>
            <div class="cpd-content">
                ${d.cpd}
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
                const currentState = parseInt(d.conditioned_state);
                
                // Only allow conditioning on the clicked node
                if (variable === d.id) {
                    const newState = currentState === value ? -1 : value;
                    
                    fetch(`/condition/${d.id}/${newState}`, { method: 'POST' })
                        .then(() => this.fetchAndUpdateState())
                        .catch(error => console.error('Condition failed:', error));
                }
            });

        // Apply initial highlighting if node is already conditioned
        this.updateTableHighlighting(d);

        // Update conditions in graph state
        if (d.evidence !== undefined) {
            this.currentGraphState.conditions[d.id] = d.evidence;
        } else {
            delete this.currentGraphState.conditions[d.id];
        }

        // Try to get samples for this node
        try {
            console.log('Fetching samples for node:', d.id);
            const response = await fetch(`/api/node_distribution/${d.id}`);
            console.log('Response status:', response.status);
            
            if (response.ok) {
                const data = await response.json();
                console.log('Received samples:', data);
                
                // Create or update the distribution plot
                const plotData = {
                    variable: d.id,
                    title: `Distribution for ${d.id}`,
                    samples: data.samples
                };
                console.log('Plot data:', plotData);
                
                // Clear the lower panel first
                const lowerPanel = d3.select(".lower-panel");
                if (!this.plotManager) {
                    console.log('Creating new PlotManager');
                    this.plotManager = new PlotManager();
                }
                
                if (!this.plotManager.plots.has('selected-node')) {
                    console.log('Creating new plot');
                    this.plotManager.createPlot('selected-node', 'distribution', plotData);
                } else {
                    console.log('Updating existing plot');
                    this.plotManager.updatePlot('selected-node', plotData);
                }
            } else {
                console.error('Failed to fetch samples:', await response.text());
            }
        } catch (error) {
            console.error('Error fetching samples:', error);
        }

        // If auto-update is enabled and we have a sampling control instance
        if (this.samplingControls && 
            this.samplingControls.getSettings().autoUpdate) {
            this.samplingControls.generateSamples();
        }
    }

    // Temporary method to generate dummy samples based on node state
    generateDummySamples(node) {
        const numSamples = 1000;
        const samples = [];
        const numStates = node.states;
        
        // If node is conditioned, generate samples mostly matching that state
        if (node.conditioned_state >= 0) {
            const prob = 0.8;  // 80% probability of matching conditioned state
            for (let i = 0; i < numSamples; i++) {
                if (Math.random() < prob) {
                    samples.push(node.conditioned_state);
                } else {
                    // Randomly choose one of the other states
                    let otherState;
                    do {
                        otherState = Math.floor(Math.random() * numStates);
                    } while (otherState === node.conditioned_state);
                    samples.push(otherState);
                }
            }
        } else {
            // If not conditioned, generate roughly uniform distribution
            for (let i = 0; i < numSamples; i++) {
                samples.push(Math.floor(Math.random() * numStates));
            }
        }
        
        return samples;
    }

    async handleSamplingRequest(settings) {
        const {
            method,
            sampleSize,
            autoUpdate,
            burnIn,
            thinning,
            randomSeed,
            cacheResults
        } = settings;

        // Prepare request data
        const requestData = {
            method,
            num_samples: sampleSize,
            conditions: this.currentGraphState.conditions,
            options: {
                burn_in: burnIn,
                thinning,
                random_seed: randomSeed,
                cache_results: cacheResults
            }
        };

        try {
            const response = await fetch('/api/sample', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`Sampling failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.currentGraphState.lastSamplingResult = result;
            
            // Update distribution plot if it exists
            if (this.distributionPlot) {
                this.distributionPlot.update(result.samples);
            }

            return {
                totalSamples: result.total_samples,
                acceptedSamples: result.accepted_samples,
                rejectedSamples: result.rejected_samples
            };
        } catch (error) {
            console.error('Error during sampling:', error);
            throw error;
        }
    }
} 