import * as d3 from 'd3';
import { PlotManager } from '/components/plot-manager';
import { PanelManager } from '/components/panel-manager';
import { calculateNodeWidth, calculateLinkEndpoints, calculateNodeClasses, calculateTableHighlightSelectors, calculateInitialNodePosition } from '/utils/graph-utils';
export class GraphVisualization {
    calculateWidth() {
        const container = document.querySelector('.graph-container');
        return container?.offsetWidth || 0;
    }
    constructor() {
        // Set up constants
        this.width = this.calculateWidth();
        this.height = window.innerHeight;
        this.nodeHeight = 30; // Fixed height for nodes
        this.textPadding = 20; // Padding on each side of text
        // Initialize simulation state
        this.simulationNodes = new Map(); // Store simulation nodes by ID
        this.simulationLinks = new Map(); // Store simulation links by ID
        this.selectedNode = null;
        this.samplingControls = null;
        // Set up SVG - clear existing content first
        d3.select(".graph-container").selectAll("svg").remove();
        this.svg = d3.select(".graph-container").append("svg")
            .attr("width", this.width)
            .attr("height", this.height);
        // Add arrow marker
        this.svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-10 -5 20 10")
            .attr("refX", 0) // Reset to 0 since we'll handle positioning in the line
            .attr("refY", 0)
            .attr("markerWidth", 20)
            .attr("markerHeight", 20)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M-10,-5L0,0L-10,5")
            .attr("fill", "#64748b");
        // Initialize simulation with gentler forces
        this.simulation = d3.forceSimulation()
            .force("link", d3.forceLink()
            .id(d => d.id)
            .distance(150)
            .strength(0.5)) // Increase link strength
            .force("charge", d3.forceManyBody().strength(-300))
            .force("collide", d3.forceCollide().radius(50))
            .force("x", d3.forceX(this.width / 2).strength(0.03))
            .force("y", d3.forceY(this.height / 2).strength(0.03))
            .alphaDecay(0.02) // Slower decay for smoother motion
            .velocityDecay(0.3) // More damping for stability
            .on("tick", () => this.tick());
        // Add containers for different elements
        this.linksGroup = this.svg.append("g");
        this.nodesGroup = this.svg.append("g");
        this.labelsGroup = this.svg.append("g");
        // Set up panel manager
        this.panelManager = new PanelManager((newWidth) => {
            this.width = newWidth;
            this.svg.attr("width", this.width);
            this.simulation.force("x", d3.forceX(this.width / 2).strength(0.05));
            this.simulation.alpha(0.3).restart();
        });
        // Start update loop
        this.startUpdateLoop();
        // Initialize sampling controls
        this.initializeSamplingControls();
        // Track current graph state
        this.currentGraphState = {
            conditions: {},
            lastSamplingResult: null
        };
    }
    async startUpdateLoop() {
        let consecutiveErrors = 0;
        const maxConsecutiveErrors = 3;
        const baseDelay = 1000;
        while (true) {
            try {
                const response = await fetch('/state');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                this.updateData(data);
                consecutiveErrors = 0; // Reset error counter on success
                await new Promise(resolve => setTimeout(resolve, baseDelay));
            }
            catch (error) {
                consecutiveErrors++;
                console.error('Error fetching data:', error);
                if (consecutiveErrors >= maxConsecutiveErrors) {
                    console.error('Too many consecutive errors, increasing delay');
                    await new Promise(resolve => setTimeout(resolve, baseDelay * 2));
                }
                else {
                    await new Promise(resolve => setTimeout(resolve, baseDelay));
                }
            }
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
                        ...node,
                        width: nodeWidth,
                        height: this.nodeHeight,
                        x: existingNode.x,
                        y: existingNode.y,
                        fx: existingNode.fx,
                        fy: existingNode.fy,
                        vx: existingNode.vx || 0,
                        vy: existingNode.vy || 0
                    });
                }
                else {
                    // For non-dragged nodes, preserve position but update state
                    const newNodeState = {
                        ...node,
                        width: nodeWidth,
                        height: this.nodeHeight
                    };
                    if (existingNode.isPinned) {
                        newNodeState.fx = existingNode.x;
                        newNodeState.fy = existingNode.y;
                    }
                    else {
                        // Preserve some momentum for smoother updates
                        newNodeState.x = existingNode.x;
                        newNodeState.y = existingNode.y;
                        newNodeState.vx = (existingNode.vx || 0) * 0.5;
                        newNodeState.vy = (existingNode.vy || 0) * 0.5;
                    }
                    Object.assign(existingNode, newNodeState);
                }
            }
            else {
                // Add new node with initial position
                const position = calculateInitialNodePosition(this.width, this.height);
                const newNode = {
                    ...node,
                    ...position,
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
            // Handle both string IDs and object references
            const sourceId = typeof link.source === 'string' ? link.source :
                'id' in link.source ? link.source.id : '';
            const targetId = typeof link.target === 'string' ? link.target :
                'id' in link.target ? link.target.id : '';
            const sourceNode = this.simulationNodes.get(sourceId);
            const targetNode = this.simulationNodes.get(targetId);
            if (sourceNode && targetNode) {
                const linkId = `${sourceId}-${targetId}`;
                this.simulationLinks.set(linkId, {
                    source: sourceNode,
                    target: targetNode,
                    id: linkId
                });
            }
        });
        // Update simulation with current data
        const nodes = Array.from(this.simulationNodes.values());
        const links = Array.from(this.simulationLinks.values());
        this.simulation
            .nodes(nodes)
            .force("link", d3.forceLink(links)
            .id((d) => d.id)
            .distance(150)
            .strength(0.5))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("collide", d3.forceCollide().radius(50))
            .force("x", d3.forceX(this.width / 2).strength(0.03))
            .force("y", d3.forceY(this.height / 2).strength(0.03))
            .alpha(0.1) // Gentler simulation restart
            .restart();
        // Update visual elements
        this.updateVisuals();
        // Update plot if we have a selected node
        if (this.selectedNode) {
            const updatedNode = newData.nodes.find(n => n.id === this.selectedNode?.id);
            if (updatedNode) {
                // Update the selected node reference
                this.selectedNode = updatedNode;
                // Update table highlighting
                this.updateTableHighlighting(updatedNode);
                // Update plot data if it exists
                if (this.plotManager && this.plotManager.hasPlot('selected-node')) {
                    // Fetch fresh distribution data for the node
                    fetch('/api/node_distribution', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            node_name: updatedNode.id,
                            codomain: 'counts'
                        })
                    })
                        .then(response => response.json())
                        .then(data => {
                        if (data.success && this.plotManager) {
                            const plotData = {
                                variable: updatedNode.id,
                                title: `Distribution for ${updatedNode.id}`,
                                x_values: data.result.x_values,
                                y_values: data.result.y_values
                            };
                            this.plotManager.updatePlot('selected-node', plotData);
                        }
                    })
                        .catch(error => console.error('Error updating plot:', error));
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
            .data(currentNodes, (d) => d.id);
        // Remove old nodes
        nodes.exit().remove();
        // Enter new nodes
        const nodeEnter = nodes.enter()
            .append("g")
            .attr("class", d => calculateNodeClasses(d).join(' '));
        // Add basic elements to new nodes
        nodeEnter
            .call(d3.drag()
            .on("start", (e, d) => this.dragstarted(e, d))
            .on("drag", (e, d) => this.dragged(e, d))
            .on("end", (e, d) => this.dragended(e, d)))
            .on("click", (e, d) => this.handleNodeClick(e, d))
            .on("dblclick", (e, d) => {
            e.preventDefault();
            e.stopPropagation();
            d.isPinned = !d.isPinned;
            if (!d.isPinned) {
                d.fx = null;
                d.fy = null;
            }
            else {
                d.fx = d.x;
                d.fy = d.y;
            }
        });
        // Add ellipse first (background)
        nodeEnter.append("ellipse")
            .attr("rx", d => (d.width || 0) / 2)
            .attr("ry", d => (d.height || 0) / 2);
        // Add text label on top
        nodeEnter.append("text")
            .attr("class", "node-label")
            .text(d => d.id);
        // Update all nodes (both new and existing)
        const allNodes = nodes.merge(nodeEnter);
        // Update node elements
        allNodes.each(function (d) {
            const node = d3.select(this);
            const isConditioned = d.conditioned_state >= 0;
            // Toggle 'conditioned' class
            node.classed("conditioned", isConditioned);
        });
        // Update links
        const links = this.linksGroup
            .selectAll("line.link")
            .data(Array.from(this.simulationLinks.values()), (d) => `${d.source.id}-${d.target.id}`);
        links.exit().remove();
        const linkEnter = links.enter()
            .append("line")
            .attr("class", "link")
            .attr("marker-end", "url(#arrowhead)")
            .style("stroke", "#999")
            .style("stroke-width", "1.5px");
        // Update all links
        links.merge(linkEnter)
            .attr("marker-end", "url(#arrowhead)")
            .style("stroke", "#999")
            .style("stroke-width", "1.5px");
        // Update state labels
        const states = this.labelsGroup
            .selectAll("text.node-states")
            .data(currentNodes, (d) => d.id);
        states.exit().remove();
        const stateEnter = states.enter()
            .append("text")
            .attr("class", "node-states");
        states.merge(stateEnter)
            .text(d => {
            const totalStates = d.states;
            const isConditioned = d.conditioned_state >= 0;
            return isConditioned ?
                `state: ${d.conditioned_state}/${totalStates}` :
                `states: ${totalStates}`;
        });
    }
    calculateNodeWidth(text) {
        // Create temporary text element to measure width
        const temp = this.svg.append("text")
            .attr("class", "node-label")
            .text(text);
        const bbox = temp.node().getBBox();
        const width = bbox.width;
        temp.remove();
        return calculateNodeWidth(width, this.textPadding);
    }
    dragstarted(event, d) {
        if (!event.active)
            this.simulation.alphaTarget(0.3).restart();
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
        if (!event.active)
            this.simulation.alphaTarget(0);
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
        const panel = d3.select(".cpd-table-panel");
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
            const value = parseInt(cell.dataset.value || "-1");
            const currentState = parseInt(d.conditioned_state.toString());
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
        }
        else {
            delete this.currentGraphState.conditions[d.id];
        }
        // Try to get distribution for this node
        try {
            console.log('Fetching distribution for node:', d.id);
            const response = await fetch('/api/node_distribution', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    node_name: d.id,
                    codomain: 'counts'
                })
            });
            console.log('Response status:', response.status);
            if (response.ok) {
                const data = await response.json();
                console.log('Received distribution data:', data);
                if (!data.success) {
                    console.error('Distribution request failed:', data.error?.message);
                    return;
                }
                // Create or update the distribution plot
                const plotData = {
                    variable: d.id,
                    title: `Distribution for ${d.id}`,
                    x_values: data.result.x_values,
                    y_values: data.result.y_values
                };
                console.log('Plot data:', plotData);
                // Set up the lower panel structure
                const lowerPanel = d3.select(".sampling-controls-panel");
                console.log('Lower panel found:', !lowerPanel.empty());
                // Ensure sampling controls don't take all space
                const samplingControls = lowerPanel.select(".sampling-controls");
                console.log('Sampling controls found:', !samplingControls.empty());
                if (!samplingControls.empty()) {
                    samplingControls
                        .style("flex", "0 0 auto")
                        .style("margin-bottom", "20px");
                }
                // Create plot manager if it doesn't exist
                if (!this.plotManager) {
                    console.log('Creating new PlotManager');
                    this.plotManager = new PlotManager();
                }
                // Create or update the plot
                if (!this.plotManager.hasPlot('selected-node')) {
                    console.log('Creating new plot');
                    this.plotManager.createPlot('selected-node', 'distribution', plotData);
                }
                else {
                    console.log('Updating existing plot');
                    this.plotManager.updatePlot('selected-node', plotData);
                }
            }
            else {
                console.error('Failed to fetch distribution:', await response.text());
            }
        }
        catch (error) {
            console.error('Error fetching distribution:', error);
        }
        // If auto-update is enabled and we have a sampling control instance
        if (this.samplingControls &&
            this.samplingControls.getSettings().autoUpdate) {
            this.samplingControls.generateSamples();
        }
    }
    async handleSamplingRequest(settings) {
        const { method, sampleSize, autoUpdate, burnIn, thinning, randomSeed, cacheResults } = settings;
        // Prepare request data
        const requestData = {
            method,
            num_samples: sampleSize,
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
            if (!result.success) {
                throw new Error(result.error?.message || 'Unknown sampling error');
            }
            this.currentGraphState.lastSamplingResult = result;
            // Update distribution plot if we have a selected node
            if (this.selectedNode && this.plotManager && this.plotManager.hasPlot('selected-node')) {
                // Fetch fresh distribution for the selected node
                const distResponse = await fetch('/api/node_distribution', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        node_name: this.selectedNode.id,
                        codomain: 'counts'
                    })
                });
                if (distResponse.ok) {
                    const distData = await distResponse.json();
                    if (distData.success) {
                        const plotData = {
                            variable: this.selectedNode.id,
                            title: `Distribution for ${this.selectedNode.id}`,
                            x_values: distData.result.x_values,
                            y_values: distData.result.y_values
                        };
                        this.plotManager.updatePlot('selected-node', plotData);
                    }
                }
            }
            return {
                totalSamples: result.result.total_samples,
                acceptedSamples: result.result.accepted_samples,
                rejectedSamples: result.result.rejected_samples
            };
        }
        catch (error) {
            console.error('Error during sampling:', error);
            throw error;
        }
    }
    async initializeSamplingControls() {
        try {
            const { SamplingControls } = await import('/components/sampling-controls');
            this.samplingControls = new SamplingControls(document.querySelector('.sampling-controls-panel'), this.handleSamplingRequest.bind(this));
        }
        catch (error) {
            console.error('Failed to initialize sampling controls:', error);
        }
    }
    updateTableHighlighting(node) {
        const table = d3.select(".cpd-table-panel .cpd-table");
        if (!table.node())
            return; // Exit if no table is displayed
        // Clear any existing highlighting
        table.selectAll("td, th").classed("state-active", false);
        const hasParents = !table.classed("no-parents");
        const selectors = calculateTableHighlightSelectors(node, hasParents);
        if (selectors) {
            table.selectAll(selectors)
                .classed("state-active", true);
        }
    }
    tick() {
        // Update links with proper node radius offset
        this.linksGroup.selectAll(".link")
            .data(Array.from(this.simulationLinks.values()))
            .join("line")
            .attr("class", "link")
            .attr("marker-end", "url(#arrowhead)")
            .style("stroke", "#64748b")
            .style("stroke-width", "1.5px")
            .each(function (d) {
            const source = d.source;
            const target = d.target;
            // Ensure we have valid coordinates before calculating endpoints
            if (typeof source.x === 'number' && typeof source.y === 'number' &&
                typeof target.x === 'number' && typeof target.y === 'number' &&
                typeof source.width === 'number' && typeof source.height === 'number' &&
                typeof target.width === 'number' && typeof target.height === 'number') {
                const endpoints = calculateLinkEndpoints({
                    x: source.x,
                    y: source.y,
                    width: source.width,
                    height: source.height
                }, {
                    x: target.x,
                    y: target.y,
                    width: target.width,
                    height: target.height
                });
                d3.select(this)
                    .attr("x1", endpoints.x1)
                    .attr("y1", endpoints.y1)
                    .attr("x2", endpoints.x2)
                    .attr("y2", endpoints.y2);
            }
        });
        // Update node groups
        this.nodesGroup.selectAll(".node")
            .attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);
        // Update labels
        this.labelsGroup.selectAll(".node-states")
            .attr("x", d => d.x || 0)
            .attr("y", d => (d.y || 0) + this.nodeHeight);
    }
}
//# sourceMappingURL=graph-visualization.js.map