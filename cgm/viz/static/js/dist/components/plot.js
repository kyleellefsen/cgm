import * as d3 from 'd3';
export class Plot {
    constructor(container, data) {
        this.container = container;
        this.data = data;
        this.margin = { top: 20, right: 20, bottom: 30, left: 40 };
    }
    get width() {
        const rect = this.container.node()?.getBoundingClientRect();
        return rect ? Math.max(100, rect.width - this.margin.left - this.margin.right) : 100;
    }
    get height() {
        const rect = this.container.node()?.getBoundingClientRect();
        // Ensure we have a minimum height and account for margins
        const containerHeight = rect?.height || 300; // Default to 300px if no height
        return Math.max(100, containerHeight - this.margin.top - this.margin.bottom);
    }
    initialize() {
        // Clear any existing content
        this.container.html("");
        // Create tooltip div if it doesn't exist
        let tooltip = d3.select('body').select('.plot-tooltip');
        if (tooltip.empty()) {
            tooltip = d3.select('body').append('div')
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
        this.svg = this.container.append("svg")
            .attr("width", totalWidth)
            .attr("height", totalHeight)
            .style("display", "block")
            .style("overflow", "visible");
        // Create plot group with margins
        this.plotGroup = this.svg.append("g")
            .attr("class", "plot-group")
            .attr("transform", `translate(${this.margin.left},${this.margin.top})`);
        // Add axes groups
        this.plotGroup.append("g")
            .attr("class", "axis x-axis")
            .attr("transform", `translate(0,${this.height})`);
        this.plotGroup.append("g")
            .attr("class", "axis y-axis");
        // Add static elements
        this.svg.append("text")
            .attr("class", "plot-title")
            .attr("text-anchor", "middle")
            .attr("x", totalWidth / 2)
            .attr("y", 25)
            .text(this.data.title || "Distribution");
        this.plotGroup.append("text")
            .attr("class", "axis-label x-label")
            .attr("text-anchor", "middle")
            .attr("x", this.width / 2)
            .attr("y", this.height + 30)
            .text("State");
        this.plotGroup.append("text")
            .attr("class", "axis-label y-label")
            .attr("text-anchor", "middle")
            .attr("transform", `translate(${-35},${this.height / 2})rotate(-90)`)
            .text("Count");
        // Initial render
        this.render();
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
}
export class DistributionPlot extends Plot {
    constructor(container, data) {
        super(container, data);
        this.margin = { top: 40, right: 20, bottom: 40, left: 50 };
        this.minHeight = 1; // Minimum bar height
        this.transitionDuration = 200;
        this.previousData = null;
        // Initialize scales
        this.xScale = d3.scaleBand()
            .padding(0.1)
            .range([0, this.width]);
        this.yScale = d3.scaleLinear()
            .range([this.height, 0]);
        // Process initial data
        const probs = this.processData(data);
        // Set initial domains
        this.xScale.domain(probs.map(d => d.state));
        this.yScale.domain([0, Math.max(1, d3.max(probs, d => d.probability) || 0)]).nice();
        // Now call initialize after scales are set up
        this.initialize();
    }
    processData(data) {
        if (!data.x_values || !data.y_values || data.x_values.length === 0) {
            return [];
        }
        // Map x_values and y_values to probability data format
        return data.x_values.map((state, i) => ({
            state,
            probability: data.y_values[i]
        }));
    }
    shouldUpdate(newData) {
        if (!this.previousData)
            return true;
        if (!newData.x_values || !newData.y_values)
            return true;
        if (!this.previousData.x_values || !this.previousData.y_values)
            return true;
        const oldProbs = this.processData(this.previousData);
        const newProbs = this.processData(newData);
        if (oldProbs.length !== newProbs.length)
            return true;
        return newProbs.some((newProb, i) => {
            const oldProb = oldProbs[i];
            return newProb.state !== oldProb.state ||
                Math.abs(newProb.probability - oldProb.probability) > 0.001;
        });
    }
    render() {
        const { x_values, y_values, title } = this.data;
        if (!x_values || !y_values || x_values.length === 0) {
            return;
        }
        // Process the data
        const probs = this.processData(this.data);
        if (!probs.length) {
            return;
        }
        // Update SVG dimensions
        const containerRect = this.container.node()?.getBoundingClientRect();
        const totalHeight = Math.max(100, containerRect?.height || 100);
        this.svg
            .attr("width", this.width + this.margin.left + this.margin.right)
            .attr("height", totalHeight);
        // Update title
        this.svg.select(".plot-title")
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
        const yAxis = d3.axisLeft(this.yScale).ticks(5).tickFormat(d3.format("d"));
        this.plotGroup.select(".x-axis").call(xAxis);
        this.plotGroup.select(".y-axis").call(yAxis);
        // Update bars with object constancy
        const bars = this.plotGroup.selectAll(".bar")
            .data(probs, d => d.state.toString());
        // ENTER new bars
        const barsEnter = bars.enter()
            .append("rect")
            .attr("class", "bar")
            .style("fill", "#4682b4")
            .style("opacity", "1")
            .attr("x", d => this.xScale(d.state) || 0)
            .attr("width", this.xScale.bandwidth())
            .attr("y", this.height)
            .attr("height", 0);
        // Add event listeners to new bars
        barsEnter
            .on("mouseover", (event, d) => {
            this.tooltip
                .style("opacity", "1")
                .html(`State ${d.state}: ${Math.round(d.probability)}`)
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
//# sourceMappingURL=plot.js.map