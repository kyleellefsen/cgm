import * as d3 from 'd3';
import { D3Selection, D3DivSelection, PlotData } from '/types';
import { Plot, DistributionPlot } from '/components/plot';

export class PlotManager {
    private container: D3DivSelection;
    private plots: Map<string, Plot>;

    constructor() {
        // Create a container for plots in the middle panel
        const plotsContainer = d3.select<HTMLDivElement, unknown>(".distribution-plot-panel .panel-content");        

        if (plotsContainer.empty()) {
            throw new Error("Distribution plots container not found");
        }
        
        // Clear any existing content
        plotsContainer.selectAll("*").remove();
            
        this.container = plotsContainer;
        this.plots = new Map();  // Store active plots
        
        // Handle panel resizing
        this.handleResize = this.handleResize.bind(this);
        window.addEventListener('resize', this.handleResize);
    }
    
    private handleResize(): void {
        // Update all plots when panel is resized
        this.plots.forEach(plot => {
            plot.render();
        });
    }
    
    public hasPlot(id: string): boolean {
        return this.plots.has(id);
    }
    
    public createPlot(id: string, type: string, data: PlotData): Plot | undefined {
        // First remove any existing plot with this id
        this.removePlot(id);
        
        // Ensure the plots container is visible
        this.container
            .style("display", "block")
            .style("visibility", "visible")
            .style("overflow", "auto");
        
        const plotContainer = this.container
            .append<HTMLDivElement>("div")
            .attr("class", "plot-container")
            .attr("id", id)
            .style("width", "100%")
            .style("height", "300px")
            .style("position", "relative")
            .style("margin-bottom", "20px")
            .style("background", "#fff")
            .style("border", "1px solid #ddd")
            .style("display", "block")
            .style("visibility", "visible") as D3DivSelection;
            
        let plot: Plot | undefined;
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
    
    public updatePlot(id: string, data: PlotData): void {
        const plot = this.plots.get(id);
        if (plot) {
            plot.update(data);
        } else {
            console.warn(`Plot ${id} not found`);
        }
    }
    
    public removePlot(id: string): void {
        const plot = this.plots.get(id);
        if (plot) {
            // Since we're using D3 selections, we need to find and remove the container
            const container = this.container.select<HTMLDivElement>(`#${id}`);
            if (!container.empty()) {
                container.remove();
            }
            this.plots.delete(id);
        }
    }
} 