import * as d3 from 'd3';
import {
    PlotData,
    Node,
    D3Selection,
    D3DivSelection,
    D3SVGSelection,
    D3SVGGSelection,
    D3SVGTextSelection,
    D3SVGRectSelection
} from '../types';

export abstract class Plot {
    protected container: D3DivSelection;
    protected data: PlotData;
    protected margin: { top: number; right: number; bottom: number; left: number };
    protected svg!: D3SVGSelection;
    protected plotGroup!: D3SVGGSelection;

    constructor(container: D3DivSelection, data: PlotData) {
        this.container = container;
        this.data = data;
        this.margin = {top: 20, right: 20, bottom: 30, left: 40};
    }

    protected initialize(): void {
        // Only setup container and basic SVG structure
        this.container.html("");
        this.svg = this.container.append<SVGElement>("svg")
            .attr("width", this.width + this.margin.left + this.margin.right)
            .attr("height", this.height + this.margin.top + this.margin.bottom)
            .style("display", "block") as D3SVGSelection;
        this.plotGroup = this.svg.append<SVGGElement>("g")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`) as D3SVGGSelection;
    }

    protected get width(): number {
        const rect = this.container.node()?.getBoundingClientRect();
        return rect ? Math.max(100, rect.width - this.margin.left - this.margin.right) : 100;
    }

    protected get height(): number {
        const rect = this.container.node()?.getBoundingClientRect();
        return rect ? rect.height - this.margin.top - this.margin.bottom : 100;
    }

    public update(data: PlotData): void {
        const hasChanged = this.shouldUpdate(data);
        if (hasChanged) {
            this.data = data;
            this.render();
        }
    }

    protected shouldUpdate(newData: PlotData): boolean {
        // Subclasses should implement their own update detection
        return true;
    }

    public abstract render(): void;
}

interface ProbabilityData {
    state: number;
    probability: number;
}

export class DistributionPlot extends Plot {
    private xScale: d3.ScaleBand<number>;
    private yScale: d3.ScaleLinear<number, number>;
    private tooltip!: D3DivSelection;
    private previousData: PlotData | null;
    private readonly minHeight: number;
    private readonly transitionDuration: number;

    constructor(container: D3DivSelection, data: PlotData) {
        super(container, data);
        this.margin = {top: 40, right: 20, bottom: 40, left: 50};
        this.minHeight = 1;  // Minimum bar height
        this.transitionDuration = 200;
        this.previousData = null;
        
        // Initialize scales
        this.xScale = d3.scaleBand<number>()
            .padding(0.1)
            .range([0, this.width]);
            
        this.yScale = d3.scaleLinear()
            .range([this.height, 0]);
            
        // Process initial data
        const probs = this.processData(data.samples || []);
        
        // Set initial domains
        this.xScale.domain(probs.map(d => d.state));
        this.yScale.domain([0, Math.max(1, d3.max(probs, d => d.probability) || 0)]).nice();
        
        // Now call initialize after scales are set up
        this.initialize();
    }

    protected initialize(): void {
        // Clear any existing content
        this.container.html("");
        
        // Create tooltip div if it doesn't exist
        let tooltip = d3.select<HTMLDivElement, unknown>('body').select<HTMLDivElement>('.plot-tooltip');
        if (tooltip.empty()) {
            tooltip = d3.select<HTMLDivElement, unknown>('body').append<HTMLDivElement>('div')
                .attr('class', 'plot-tooltip')
                .style('position', 'absolute')
                .style('opacity', '0')
                .style('background', 'white')
                .style('border', '1px solid black')
                .style('padding', '5px')
                .style('pointer-events', 'none');
        }
        this.tooltip = tooltip;
        
        // Create SVG container with explicit dimensions
        const containerRect = this.container.node()?.getBoundingClientRect();
        const totalWidth = Math.max(100, containerRect?.width || 100);
        const totalHeight = Math.max(100, containerRect?.height || 100);
        
        this.svg = this.container.append<SVGElement>("svg")
            .attr("width", totalWidth)
            .attr("height", totalHeight)
            .style("display", "block")
            .style("overflow", "visible") as D3SVGSelection;
            
        // Create plot group with margins
        this.plotGroup = this.svg.append<SVGGElement>("g")
            .attr("class", "plot-group")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`) as D3SVGGSelection;
            
        // Add axes groups
        this.plotGroup.append<SVGGElement>("g")
            .attr("class", "axis x-axis")
            .attr("transform", `translate(0,${this.height})`);
            
        this.plotGroup.append<SVGGElement>("g")
            .attr("class", "axis y-axis");
            
        // Add static elements
        this.svg.append<SVGTextElement>("text")
            .attr("class", "plot-title")
            .attr("text-anchor", "middle")
            .attr("x", totalWidth/2)
            .attr("y", 25)
            .text(this.data.title || "Distribution");

        this.plotGroup.append<SVGTextElement>("text")
            .attr("class", "axis-label x-label")
            .attr("text-anchor", "middle")
            .attr("x", this.width/2)
            .attr("y", this.height + 30)
            .text("State");

        this.plotGroup.append<SVGTextElement>("text")
            .attr("class", "axis-label y-label")
            .attr("text-anchor", "middle")
            .attr("transform", `translate(${-35},${this.height/2})rotate(-90)`)
            .text("Probability");
            
        // Initial render
        this.render();
    }

    private processData(samples: number[]): ProbabilityData[] {
        if (!samples || !samples.length) return [];
        
        // Count occurrences of each state
        const counts = new Map<number, number>();
        samples.forEach(s => counts.set(s, (counts.get(s) || 0) + 1));
        
        // Convert to probability
        const probs = Array.from(counts.entries()).map(([state, count]) => ({
            state,
            probability: count / samples.length
        }));
        
        // Sort by state
        return probs.sort((a, b) => a.state - b.state);
    }

    protected shouldUpdate(newData: PlotData): boolean {
        if (!this.previousData) return true;
        if (!newData || !newData.samples) return true;
        
        const oldProbs = this.processData(this.previousData.samples || []);
        const newProbs = this.processData(newData.samples || []);
        
        if (oldProbs.length !== newProbs.length) return true;
        
        return newProbs.some((newProb, i) => {
            const oldProb = oldProbs[i];
            return newProb.state !== oldProb.state || 
                   Math.abs(newProb.probability - oldProb.probability) > 0.001;
        });
    }

    public render(): void {
        console.log('Rendering plot with data:', this.data);
        console.log('Plot dimensions:', {
            width: this.width,
            height: this.height,
            margin: this.margin
        });
        
        const {samples, title} = this.data;
        if (!samples || !samples.length) return;
        
        // Process the data
        const probs = this.processData(samples);
        if (!probs.length) return;
        
        // Update SVG dimensions
        const containerRect = this.container.node()?.getBoundingClientRect();
        const totalHeight = Math.max(100, containerRect?.height || 100);
        
        this.svg
            .attr("width", this.width + this.margin.left + this.margin.right)
            .attr("height", totalHeight);
            
        // Update title
        this.svg.select<SVGTextElement>(".plot-title")
            .text(title || "Distribution");
        
        // Update scales
        this.xScale
            .range([0, this.width])
            .domain(probs.map(d => d.state));
            
        this.yScale
            .range([this.height, 0])
            .domain([0, Math.max(1, d3.max(probs, d => d.probability) || 0)])
            .nice();
            
        // Update axes
        const xAxis = d3.axisBottom(this.xScale).tickFormat(d => `State ${String(d)}`);
        const yAxis = d3.axisLeft(this.yScale).ticks(5).tickFormat(d3.format(".1%"));
        
        this.plotGroup.select<SVGGElement>(".x-axis").call(xAxis);
        this.plotGroup.select<SVGGElement>(".y-axis").call(yAxis);
        
        // Update bars with object constancy
        const bars = this.plotGroup.selectAll<SVGRectElement, ProbabilityData>(".bar")
            .data(probs, d => d.state.toString());
            
        // ENTER new bars
        const barsEnter = bars.enter()
            .append<SVGRectElement>("rect")
            .attr("class", "bar")
            .style("fill", "#4682b4")
            .style("opacity", "1")
            .attr("x", d => this.xScale(d.state) || 0)
            .attr("width", this.xScale.bandwidth())
            .attr("y", this.height)
            .attr("height", 0);
            
        // Add event listeners to new bars
        barsEnter
            .on("mouseover", (event: MouseEvent, d: ProbabilityData) => {
                this.tooltip
                    .style("opacity", "1")
                    .html(`State ${d.state}: ${(d.probability * 100).toFixed(0)}%`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            })
            .on("mouseout", () => {
                this.tooltip.style("opacity", "0");
            });
            
        // EXIT old bars
        bars.exit().remove();
            
        // UPDATE + ENTER: merge and transition
        const allBars = bars.merge(barsEnter);
        
        // Update all bars with transition
        allBars.transition()
            .duration(this.transitionDuration)
            .attr("x", d => this.xScale(d.state) || 0)
            .attr("width", this.xScale.bandwidth())
            .attr("y", d => this.yScale(d.probability))
            .attr("height", d => Math.max(this.minHeight, this.height - this.yScale(d.probability)))
            .style("opacity", "1");
    }
} 