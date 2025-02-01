class Plot {
    constructor(container, data) {
        this.container = container;
        this.data = data;
        this.margin = {top: 20, right: 20, bottom: 30, left: 40};
        this.initialize();
    }

    initialize() {
        // Clear any existing content
        this.container.html("");
        
        // Create SVG container
        this.svg = this.container.append("svg")
            .style("width", "100%")
            .style("height", "100%");
            
        // Create plot group with margins
        this.plotGroup = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
            
        // Add clip path to prevent plotting outside bounds
        this.svg.append("defs").append("clipPath")
            .attr("id", "plot-area")
            .append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", this.width)
            .attr("height", this.height);
    }

    get width() {
        return this.container.node().offsetWidth - this.margin.left - this.margin.right;
    }

    get height() {
        return this.container.node().offsetHeight - this.margin.top - this.margin.bottom;
    }

    update(data) {
        this.data = data;
        this.render();
    }

    render() {
        // Override in subclasses
        console.warn("Base Plot render() called - override in subclass");
    }
}

class DistributionPlot extends Plot {
    constructor(container, data) {
        super(container, data);
        this.margin = {top: 40, right: 20, bottom: 40, left: 50};  // Increased margins for labels
        this.xScale = d3.scaleLinear();
        this.yScale = d3.scaleLinear();
        this.histogram = d3.histogram();
    }

    initialize() {
        // Clear any existing content
        this.container.html("");
        
        // Create SVG container with explicit dimensions
        this.svg = this.container.append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .style("overflow", "visible");  // Allow labels to overflow
            
        // Create plot group with margins
        this.plotGroup = this.svg.append("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
            
        // Add axes
        this.xAxis = this.plotGroup.append("g")
            .attr("class", "axis x-axis")
            .attr("transform", `translate(0,${this.height})`);
            
        this.yAxis = this.plotGroup.append("g")
            .attr("class", "axis y-axis");
            
        // Add title
        this.title = this.svg.append("text")
            .attr("class", "plot-title")
            .attr("text-anchor", "middle")
            .attr("x", this.margin.left + this.width/2)
            .attr("y", 25)
            .text(this.data.title || "Distribution");

        // Add axis labels
        this.plotGroup.append("text")
            .attr("class", "axis-label")
            .attr("text-anchor", "middle")
            .attr("x", this.width/2)
            .attr("y", this.height + 30)
            .text("Value");

        this.plotGroup.append("text")
            .attr("class", "axis-label")
            .attr("text-anchor", "middle")
            .attr("transform", `translate(${-35},${this.height/2})rotate(-90)`)
            .text("Frequency");
    }

    render() {
        const {samples, variable} = this.data;
        if (!samples || !samples.length) return;

        // Update scales
        this.xScale
            .domain([0, d3.max(samples)])
            .range([0, this.width]);
            
        // Generate histogram data
        const bins = this.histogram
            .domain(this.xScale.domain())
            .thresholds(this.xScale.ticks(10))  // Reduced number of bins
            (samples);
            
        // Update y scale based on histogram
        this.yScale
            .domain([0, d3.max(bins, d => d.length)])
            .range([this.height, 0]);
            
        // Update axes with nicer ticks
        this.xAxis.call(
            d3.axisBottom(this.xScale)
                .ticks(5)  // Reduced number of ticks
                .tickFormat(d3.format(".1f"))
        );
        this.yAxis.call(
            d3.axisLeft(this.yScale)
                .ticks(5)
                .tickFormat(d3.format("d"))
        );
        
        // Draw bars
        const bars = this.plotGroup.selectAll(".bar")
            .data(bins);
            
        // Remove old bars
        bars.exit().remove();
        
        // Add new bars
        const barsEnter = bars.enter()
            .append("rect")
            .attr("class", "bar");
            
        // Update all bars
        bars.merge(barsEnter)
            .transition()
            .duration(200)
            .attr("x", d => this.xScale(d.x0))
            .attr("y", d => this.yScale(d.length))
            .attr("width", d => Math.max(0, this.xScale(d.x1) - this.xScale(d.x0) - 1))
            .attr("height", d => this.height - this.yScale(d.length));
    }
}

class PlotManager {
    constructor() {
        this.container = d3.select(".lower-panel");
        this.plots = new Map();  // Store active plots
        
        // Initialize with empty state
        this.container.html(`
            <div class="panel-header">Plots</div>
            <div class="plots-container"></div>
        `);
        
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
        const plotContainer = this.container.select(".plots-container")
            .append("div")
            .attr("class", "plot-container")
            .attr("id", `plot-${id}`);
            
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
        
        // Start update loop
        this.startUpdateLoop();
        
        this.conditioned_vars = {};
        
        // Initialize plot manager
        this.plotManager = new PlotManager();
        
        // Add test plot with dummy data - create a normal distribution
        const normalData = Array.from({length: 1000}, () => {
            // Box-Muller transform to generate normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            return z + 2;  // Shift mean to 2
        });
        
        const dummyData = {
            variable: 'test',
            title: 'Test Distribution',
            samples: normalData
        };
        this.plotManager.createPlot('test', 'distribution', dummyData);
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

        // Update any open CPD table
        if (this.selectedNode) {
            const updatedNode = newData.nodes.find(n => n.id === this.selectedNode.id);
            if (updatedNode) {
                this.handleNodeClick(null, updatedNode);
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

    handleNodeClick(event, d) {
        if (event) {
            event.preventDefault();
            event.stopPropagation();
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
    }
} 