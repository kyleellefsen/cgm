import * as d3 from 'd3';
import { 
    D3Selection, 
    D3DivSelection, 
    D3SVGSelection,
    D3SVGGSelection,
    SimulationNode,
    SimulationLink,
    GraphState,
    SamplingSettings,
    SamplingResult,
    NodeSelection,
    LinkSelection,
    PanelDimensions
} from '../types';
import { PlotManager } from '/components/plot-manager';
import { SamplingControls } from '/components/sampling-controls';
import { PanelManager } from '/components/panel-manager';

interface HTMLElementWithStyle extends HTMLElement {
    style: CSSStyleDeclaration;
    offsetWidth: number;
    offsetHeight: number;
}

export class GraphVisualization {
    private width: number;
    private height: number;
    private nodeHeight: number;
    private textPadding: number;
    private simulationNodes: Map<string, SimulationNode>;
    private simulationLinks: Map<string, SimulationLink>;
    private selectedNode: SimulationNode | null;
    private svg!: D3SVGSelection;
    private simulation: d3.Simulation<SimulationNode, SimulationLink>;
    private linksGroup!: D3SVGGSelection;
    private nodesGroup!: D3SVGGSelection;
    private labelsGroup!: D3SVGGSelection;
    private plotManager?: PlotManager;
    private currentGraphState!: GraphState;
    private samplingControls!: SamplingControls | null;
    private panelManager: PanelManager;

    constructor(panelManager: PanelManager) {
        this.panelManager = panelManager;
        
        // Set up constants
        this.width = this.calculateWidth();
        this.height = window.innerHeight;
        this.nodeHeight = 30;  // Fixed height for nodes
        this.textPadding = 20; // Padding on each side of text
        
        // Initialize simulation state
        this.simulationNodes = new Map(); // Store simulation nodes by ID
        this.simulationLinks = new Map(); // Store simulation links by ID
        this.selectedNode = null;
        this.samplingControls = null;
        
        // Set up SVG - clear existing content first
        d3.select(".graph-container").selectAll("svg").remove();
        this.svg = d3.select<HTMLElement, unknown>(".graph-container").append<SVGSVGElement>("svg")
            .attr("width", this.width)
            .attr("height", this.height) as unknown as D3SVGSelection;
            
        // Add arrow marker
        this.svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-10 -5 20 10")
            .attr("refX", 0)  // Reset to 0 since we'll handle positioning in the line
            .attr("refY", 0)
            .attr("markerWidth", 20)
            .attr("markerHeight", 20)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M-10,-5L0,0L-10,5")
            .attr("fill", "#64748b");

        // Initialize simulation with gentler forces
        this.simulation = d3.forceSimulation<SimulationNode>()
            .force("link", d3.forceLink<SimulationNode, SimulationLink>()
                .id(d => d.id)
                .distance(150)
                .strength(0.5))  // Increase link strength
            .force("charge", d3.forceManyBody().strength(-300))
            .force("collide", d3.forceCollide().radius(50))
            .force("x", d3.forceX<SimulationNode>(this.width / 2).strength(0.03))
            .force("y", d3.forceY<SimulationNode>(this.height / 2).strength(0.03))
            .alphaDecay(0.02)  // Slower decay for smoother motion
            .velocityDecay(0.3)  // More damping for stability
            .on("tick", () => this.tick());
            
        // Add containers for different elements
        this.linksGroup = this.svg.append<SVGGElement>("g") as unknown as D3SVGGSelection;
        this.nodesGroup = this.svg.append<SVGGElement>("g") as unknown as D3SVGGSelection;
        this.labelsGroup = this.svg.append<SVGGElement>("g") as unknown as D3SVGGSelection;
        
        // Start update loop
        this.startUpdateLoop();
        
        // Initialize sampling controls
        this.initializeSamplingControls();
        
        // Track current graph state
        this.currentGraphState = {
            conditions: {},
            lastSamplingResult: null
        };

        // Set up panel resize handlers
        this.setupPanelHandlers();
    }

    private setupPanelHandlers(): void {
        // Handle graph container resizing
        this.panelManager.onResize((event) => {
            if (event.panelId === 'graph-container') {
                this.width = event.dimensions.width;
                this.height = event.dimensions.height;
                this.svg.attr("width", this.width)
                    .attr("height", this.height);
                this.simulation.force("x", d3.forceX(this.width / 2).strength(0.03))
                    .force("y", d3.forceY(this.height / 2).strength(0.03));
                this.simulation.alpha(0.3).restart();
            }
        });
    }

    calculateWidth(): number {
        const container = document.querySelector('.graph-container') as HTMLElementWithStyle;
        return container?.offsetWidth || 0;
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
                consecutiveErrors = 0;  // Reset error counter on success
                await new Promise(resolve => setTimeout(resolve, baseDelay));
            } catch (error) {
                consecutiveErrors++;
                console.error('Error fetching data:', error);
                
                if (consecutiveErrors >= maxConsecutiveErrors) {
                    console.error('Too many consecutive errors, increasing delay');
                    await new Promise(resolve => setTimeout(resolve, baseDelay * 2));
                } else {
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

    updateData(newData: { nodes: SimulationNode[]; links: SimulationLink[] }) {
        // console.log('Received new data:', newData);
        
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
                        vx: existingNode.vx || 0,
                        vy: existingNode.vy || 0
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
                        newNodeState.vx = (existingNode.vx || 0) * 0.5;
                        newNodeState.vy = (existingNode.vy || 0) * 0.5;
                    }
                    
                    Object.assign(existingNode, newNodeState);
                }
            } else {
                // Add new node with initial position
                const newNode: SimulationNode = {
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
            .force("link", d3.forceLink<SimulationNode, SimulationLink>(links)
                .id((d: SimulationNode) => d.id)
                .distance(150)
                .strength(0.5))
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
            .selectAll<SVGGElement, SimulationNode>("g.node")
            .data(currentNodes, (d: SimulationNode) => d.id);
            
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
            .call(d3.drag<SVGGElement, SimulationNode>()
                .on("start", (e,d) => this.dragstarted(e,d))
                .on("drag", (e,d) => this.dragged(e,d))
                .on("end", (e,d) => this.dragended(e,d)))
            .on("click", (e,d) => this.handleNodeClick(e,d))
            .on("dblclick", (e,d) => {
                e.preventDefault();
                e.stopPropagation();
                d.isPinned = !d.isPinned;
                if (!d.isPinned) {
                    d.fx = null;
                    d.fy = null;
                } else {
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
        const allNodes = nodes.merge(nodeEnter as any);
        
        // Update node elements
        allNodes.each(function(d) {
            const node = d3.select(this) as d3.Selection<SVGGElement, SimulationNode, null, undefined>;
            const isConditioned = d.conditioned_state >= 0;
            
            // Toggle 'conditioned' class
            node.classed("conditioned", isConditioned);
        });
        
        // Update links
        const links = this.linksGroup
            .selectAll<SVGLineElement, SimulationLink>("line.link")
            .data(Array.from(this.simulationLinks.values()), 
                (d: SimulationLink) => `${d.source.id}-${d.target.id}`);
            
        links.exit().remove();
        
        const linkEnter = links.enter()
            .append("line")
            .attr("class", "link")
            .attr("marker-end", "url(#arrowhead)")
            .style("stroke", "#999")
            .style("stroke-width", "1.5px");
            
        // Update all links
        links.merge(linkEnter as any)
            .attr("marker-end", "url(#arrowhead)")
            .style("stroke", "#999")
            .style("stroke-width", "1.5px");
        
        // Update state labels
        const states = this.labelsGroup
            .selectAll<SVGTextElement, SimulationNode>("text.node-states")
            .data(currentNodes, (d: SimulationNode) => d.id);
            
        states.exit().remove();
        
        const stateEnter = states.enter()
            .append("text")
            .attr("class", "node-states");
            
        states.merge(stateEnter as any)
            .text(d => {
                const totalStates = d.states;
                const isConditioned = d.conditioned_state >= 0;
                return isConditioned ? 
                    `state: ${d.conditioned_state}/${totalStates}` : 
                    `states: ${totalStates}`;
            });
    }

    calculateNodeWidth(text: string): number {
        // Create temporary text element to measure width
        const temp = this.svg.append("text")
            .attr("class", "node-label")
            .text(text);
        const bbox = (temp.node() as SVGTextElement).getBBox();
        const width = bbox.width;
        temp.remove();
        return width + this.textPadding * 2; // Add padding on both sides
    }

    dragstarted(event: d3.D3DragEvent<SVGGElement, SimulationNode, unknown>, d: SimulationNode) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        d.isDragging = true;
        event.sourceEvent.stopPropagation();
    }

    dragged(event: d3.D3DragEvent<SVGGElement, SimulationNode, unknown>, d: SimulationNode) {
        d.fx = event.x;
        d.fy = event.y;
        event.sourceEvent.stopPropagation();
    }

    dragended(event: d3.D3DragEvent<SVGGElement, SimulationNode, unknown>, d: SimulationNode) {
        if (!event.active) this.simulation.alphaTarget(0);
        // Only keep position fixed if the node was explicitly pinned
        if (!d.isPinned) {
            d.fx = null;
            d.fy = null;
        }
        d.isDragging = false;
        event.sourceEvent.stopPropagation();
    }

    async handleNodeClick(event: MouseEvent | null, d: SimulationNode) {
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

        // Store the selected node for later updates
        this.selectedNode = d;

        // Update CPD panel
        if (!d.cpd) {
            this.panelManager.setPanelContent('cpd-panel', `
                <div class="panel-placeholder">No CPD available for ${d.id}</div>
            `);
            return;
        }

        // Set the CPD content
        this.panelManager.setPanelContent('cpd-panel', `
            <div class="cpd-content">${d.cpd}</div>
        `);

        // Get the panel and check for table
        const panel = this.panelManager.getPanel('cpd-panel');
        if (!panel) return;

        const table = panel.querySelector<HTMLTableElement>('.cpd-table');
        if (!table) return;

        // Determine if this node has parents by checking for multiple rows
        const hasParents = table.querySelectorAll('tbody tr').length > 1;
        table.classList.toggle('no-parents', !hasParents);

        // Add click handlers for state cells
        panel.querySelectorAll<HTMLElement>('[data-variable][data-value]')
            .forEach(cell => {
                cell.addEventListener('click', () => {
                    const variable = cell.dataset.variable;
                    const value = parseInt(cell.dataset.value || "-1");
                    const currentState = parseInt(d.conditioned_state.toString());
                    
                    if (variable === d.id) {
                        const newState = currentState === value ? -1 : value;
                        
                        fetch(`/condition/${d.id}/${newState}`, { method: 'POST' })
                            .then(() => this.fetchAndUpdateState())
                            .catch(error => console.error('Condition failed:', error));
                    }
                });
            });

        // Apply initial highlighting if node is already conditioned
        this.updateTableHighlighting(d);

        // Update conditions in graph state
        if (d.evidence !== undefined) {
            this.currentGraphState.conditions[d.id] = d.evidence;
        } else {
            delete this.currentGraphState.conditions[d.id];
        }

        // Try to get distribution for this node
        try {
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
            
            if (response.ok) {
                const data = await response.json();
                
                if (!data.success) {
                    console.error('Distribution request failed:', data.error?.message);
                    return;
                }
                
                const plotData = {
                    variable: d.id,
                    title: `Distribution for ${d.id}`,
                    x_values: data.result.x_values,
                    y_values: data.result.y_values
                };
                
                // Create plot manager if it doesn't exist
                if (!this.plotManager) {
                    this.plotManager = new PlotManager();
                }
                
                // Create or update the plot
                if (!this.plotManager.hasPlot('selected-node')) {
                    this.plotManager.createPlot('selected-node', 'distribution', plotData);
                } else {
                    this.plotManager.updatePlot('selected-node', plotData);
                }
            } else {
                console.error('Failed to fetch distribution:', await response.text());
            }
        } catch (error) {
            console.error('Error fetching distribution:', error);
        }

        // If auto-update is enabled and we have a sampling control instance
        if (this.samplingControls && 
            this.samplingControls.getSettings().autoUpdate) {
            this.samplingControls.generateSamples();
        }
    }

    async handleSamplingRequest(settings: SamplingSettings): Promise<SamplingResult> {
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
        } catch (error) {
            console.error('Error during sampling:', error);
            throw error;
        }
    }

    async initializeSamplingControls() {
        try {
            const { SamplingControls } = await import('/components/sampling-controls');
            const panel = this.panelManager.getPanel('sampling-controls');
            if (panel) {
                const contentContainer = panel.querySelector('.panel-content');
                if (contentContainer) {
                    this.samplingControls = new SamplingControls(
                        contentContainer as HTMLElement,
                        this.handleSamplingRequest.bind(this)
                    );
                }
            }
        } catch (error) {
            console.error('Failed to initialize sampling controls:', error);
        }
    }

    updateTableHighlighting(node: SimulationNode) {
        const panel = this.panelManager.getPanel('cpd-panel');
        if (!panel) return;

        const table = panel.querySelector<HTMLTableElement>('.cpd-table');
        if (!table) return;

        // Clear any existing highlighting
        table.querySelectorAll('td, th').forEach(cell => {
            cell.classList.remove('state-active');
        });

        const currentState = parseInt(node.conditioned_state.toString());
        if (currentState === -1) return;  // No highlighting needed

        const hasParents = !table.classList.contains('no-parents');
        
        if (hasParents) {
            // For nodes with parents, highlight cells and header showing the current state
            table.querySelectorAll(`td[data-variable="${node.id}"][data-value="${currentState}"], 
                                  th[data-variable="${node.id}"][data-value="${currentState}"]`)
                .forEach(cell => cell.classList.add('state-active'));
        } else {
            // For nodes without parents, highlight the column corresponding to the state
            // Add 1 to account for the label column
            const stateColumn = currentState + 1;
            table.querySelectorAll(`td:nth-child(${stateColumn}), 
                                  th:nth-child(${stateColumn})`)
                .forEach(cell => cell.classList.add('state-active'));
        }
    }

    tick() {
        // Update links with proper node radius offset
        this.linksGroup.selectAll<SVGLineElement, SimulationLink>(".link")
            .data(Array.from(this.simulationLinks.values()))
            .join("line")
            .attr("class", "link")
            .attr("marker-end", "url(#arrowhead)")
            .style("stroke", "#64748b")
            .style("stroke-width", "1.5px")
            .attr("x1", d => {
                const dx = (d.target.x || 0) - (d.source.x || 0);
                const dy = (d.target.y || 0) - (d.source.y || 0);
                const angle = Math.atan2(dy, dx);
                return (d.source.x || 0) + Math.cos(angle) * ((d.source.width || 0) / 2);
            })
            .attr("y1", d => {
                const dx = (d.target.x || 0) - (d.source.x || 0);
                const dy = (d.target.y || 0) - (d.source.y || 0);
                const angle = Math.atan2(dy, dx);
                return (d.source.y || 0) + Math.sin(angle) * ((d.source.height || 0) / 2);
            })
            .attr("x2", d => {
                const dx = (d.target.x || 0) - (d.source.x || 0);
                const dy = (d.target.y || 0) - (d.source.y || 0);
                const angle = Math.atan2(dy, dx);
                const targetRadius = ((d.target.width || 0) / 2);
                return (d.target.x || 0) - Math.cos(angle) * targetRadius;  // Remove the offset
            })
            .attr("y2", d => {
                const dx = (d.target.x || 0) - (d.source.x || 0);
                const dy = (d.target.y || 0) - (d.source.y || 0);
                const angle = Math.atan2(dy, dx);
                const targetRadius = ((d.target.height || 0) / 2);
                return (d.target.y || 0) - Math.sin(angle) * targetRadius;  // Remove the offset
            });

        // Update node groups
        this.nodesGroup.selectAll<SVGGElement, SimulationNode>(".node")
            .attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);

        // Update labels
        this.labelsGroup.selectAll<SVGTextElement, SimulationNode>(".node-states")
            .attr("x", d => d.x || 0)
            .attr("y", d => (d.y || 0) + this.nodeHeight);
    } 
}
