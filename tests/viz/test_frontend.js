/**
 * @jest-environment jsdom
 */

import * as d3 from 'd3';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { jest } from '@jest/globals';
import { GraphVisualization, PlotManager, DistributionPlot } from '../../cgm/viz/static/js/viz-graph.js';
import { mockD3Transition } from '../../jest.setup.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Mock D3 transitions
jest.mock('d3', async () => {
    const actual = await jest.importActual('d3');
    return {
        ...actual,
        transition: () => mockD3Transition
    };
});

// Mock fetch for API calls
global.fetch = jest.fn();

// Mock SamplingControls class
global.SamplingControls = class {
    constructor() {
        this.getSettings = () => ({
            autoUpdate: false,
            method: 'forward',
            sampleSize: 1000,
            burnIn: 100,
            thinning: 1,
            randomSeed: 42,
            cacheResults: true
        });
    }
};

// Mock tooltip div
beforeAll(() => {
    const tooltipDiv = document.createElement('div');
    tooltipDiv.className = 'plot-tooltip';
    document.body.appendChild(tooltipDiv);
});

describe('Distribution Plot Tests', () => {
    let container;
    let plot;
    
    beforeEach(() => {
        // Set up a container div with specific dimensions
        container = d3.select(document.createElement('div'))
            .style('width', '500px')
            .style('height', '300px');
            
        // Mock getBoundingClientRect for the container
        container.node().getBoundingClientRect = () => ({
            width: 500,
            height: 300,
            top: 0,
            left: 0,
            bottom: 300,
            right: 500
        });
            
        // Create test data
        const testData = {
            samples: [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],  // 6 zeros, 4 ones
            title: 'Test Distribution'
        };
        
        // Create the plot
        plot = new DistributionPlot(container, testData);
        
        // Force a render without transitions
        plot.render();
    });
    
    test('plot is created with correct structure', () => {
        // Check SVG exists
        const svg = container.select('svg');
        expect(svg.empty()).toBeFalsy();
        
        // Check basic elements exist
        expect(svg.select('.plot-title').empty()).toBeFalsy();
        expect(svg.select('.x-axis').empty()).toBeFalsy();
        expect(svg.select('.y-axis').empty()).toBeFalsy();
    });
    
    test('bars are created with correct heights', (done) => {
        // Force another render to ensure bars are created
        plot.render();
        
        // Debug: Check plot dimensions
        console.log('Plot dimensions:', {
            width: plot.width,
            height: plot.height,
            margin: plot.margin
        });
        
        // Debug: Check scales
        console.log('X scale:', {
            domain: plot.xScale.domain(),
            range: plot.xScale.range(),
            bandwidth: plot.xScale.bandwidth()
        });
        console.log('Y scale:', {
            domain: plot.yScale.domain(),
            range: plot.yScale.range()
        });
        
        // Wait for transitions to complete
        setTimeout(() => {
            // Get all bars
            const bars = container.selectAll('.bar');
            expect(bars.size()).toBe(2);  // Should have 2 bars for states 0 and 1
            
            // Get the bars data
            const barsData = [];
            bars.each(function() {
                const bar = d3.select(this);
                const datum = bar.datum();
                barsData.push({
                    height: parseFloat(bar.attr('height')),
                    y: parseFloat(bar.attr('y')),
                    state: datum.state,
                    probability: datum.probability,
                    rawAttrs: {
                        x: bar.attr('x'),
                        y: bar.attr('y'),
                        width: bar.attr('width'),
                        height: bar.attr('height')
                    }
                });
            });
            
            // Debug: Log the bars data
            console.log('Bars data:', barsData);
            
            // Sort bars by state to ensure consistent order
            barsData.sort((a, b) => a.state - b.state);
            
            // Check probabilities (should be 0.6 and 0.4)
            expect(barsData[0].probability).toBeCloseTo(0.6, 2);  // State 0: 6/10
            expect(barsData[1].probability).toBeCloseTo(0.4, 2);  // State 1: 4/10
            
            // Check that bars have non-zero heights
            expect(barsData[0].height).toBeGreaterThan(0);
            expect(barsData[1].height).toBeGreaterThan(0);
            
            // Check relative heights (should be proportional to probabilities)
            const heightRatio = barsData[0].height / barsData[1].height;
            expect(heightRatio).toBeCloseTo(1.5, 1);  // 0.6/0.4 = 1.5
            
            done();
        }, 300);  // Wait longer than the transition duration (200ms)
    });
});

describe('Plot Manager Tests', () => {
    let container;
    let plotManager;
    
    beforeEach(() => {
        // Create required DOM structure
        document.body.innerHTML = `
            <div class="lower-panel">
                <div class="sampling-controls"></div>
                <div class="plots-container"></div>
            </div>
        `;
        
        // Initialize plot manager
        plotManager = new PlotManager();
    });
    
    afterEach(() => {
        document.body.innerHTML = '';
    });
    
    test('creates new plot correctly', () => {
        const plotId = 'test-plot';
        const plotData = {
            samples: [0, 1, 0, 1, 0],
            title: 'Test Plot'
        };
        
        const plot = plotManager.createPlot(plotId, 'distribution', plotData);
        
        expect(plot).toBeDefined();
        expect(plot instanceof DistributionPlot).toBeTruthy();
        
        // Check if plot was added to DOM
        const plotElement = document.querySelector(`#${plotId}`);
        expect(plotElement).toBeTruthy();
    });
    
    test('updates existing plot', () => {
        const plotId = 'test-plot';
        const initialData = {
            samples: [0, 1, 0, 1, 0],
            title: 'Test Plot'
        };
        
        const plot = plotManager.createPlot(plotId, 'distribution', initialData);
        const updateSpy = jest.spyOn(plot, 'update');
        
        const newData = {
            samples: [1, 1, 1, 0, 0],
            title: 'Updated Plot'
        };
        
        plotManager.updatePlot(plotId, newData);
        
        expect(updateSpy).toHaveBeenCalledWith(newData);
    });
    
    test('removes plot correctly', () => {
        const plotId = 'test-plot';
        const plotData = {
            samples: [0, 1, 0, 1, 0],
            title: 'Test Plot'
        };
        
        plotManager.createPlot(plotId, 'distribution', plotData);
        const plotElement = document.querySelector(`#${plotId}`);
        expect(plotElement).toBeTruthy();
        
        plotManager.removePlot(plotId);
        
        // Verify plot is removed from manager and DOM
        expect(plotManager.plots.has(plotId)).toBeFalsy();
        expect(document.querySelector(`#${plotId}`)).toBeNull();
    });
});

describe('Distribution Plot Extended Tests', () => {
    let container;
    
    beforeEach(() => {
        container = d3.select(document.createElement('div'))
            .style('width', '500px')
            .style('height', '300px');
            
        container.node().getBoundingClientRect = () => ({
            width: 500,
            height: 300,
            top: 0,
            left: 0,
            bottom: 300,
            right: 500
        });
    });
    
    test('handles empty data gracefully', () => {
        const testData = {
            samples: [],
            title: 'Empty Distribution'
        };
        
        const plot = new DistributionPlot(container, testData);
        plot.render();
        
        // Should create empty plot without errors
        expect(container.select('svg').empty()).toBeFalsy();
        expect(container.select('.plot-title').text()).toBe('Empty Distribution');
    });
    
    test('updates data and transitions correctly', (done) => {
        const initialData = {
            samples: [0, 0, 0, 1, 1],
            title: 'Initial Distribution'
        };
        
        const plot = new DistributionPlot(container, initialData);
        plot.render();
        
        // Get initial bar heights
        const initialHeights = [];
        container.selectAll('.bar').each(function() {
            initialHeights.push(d3.select(this).attr('height'));
        });
        
        // Update with new data
        const newData = {
            samples: [1, 1, 1, 1, 0],
            title: 'Updated Distribution'
        };
        
        plot.update(newData);
        
        // Wait for transition to complete
        setTimeout(() => {
            const newHeights = [];
            container.selectAll('.bar').each(function() {
                newHeights.push(d3.select(this).attr('height'));
            });
            
            // Heights should be different after update
            expect(newHeights).not.toEqual(initialHeights);
            
            // Title should be updated
            expect(container.select('.plot-title').text()).toBe('Updated Distribution');
            
            done();
        }, 1000); // Increase timeout to ensure transition completes
    }, 10000); // Add test timeout
    
    test('handles window resize', () => {
        const testData = {
            samples: [0, 1, 0, 1, 0],
            title: 'Resize Test'
        };
        
        const plot = new DistributionPlot(container, testData);
        plot.render();
        
        const initialWidth = plot.width;
        const initialHeight = plot.height;
        
        // Mock container resize
        container.style('width', '600px');
        container.style('height', '400px');
        container.node().getBoundingClientRect = () => ({
            width: 600,
            height: 400,
            top: 0,
            left: 0,
            bottom: 400,
            right: 600
        });
        
        // Trigger resize by calling render
        plot.render();
        
        // SVG should match new container size
        const svg = container.select('svg');
        expect(parseInt(svg.attr('width'))).toBe(600);
        expect(parseInt(svg.attr('height'))).toBe(400);
        
        // Plot dimensions should be updated
        expect(plot.width).toBeGreaterThan(initialWidth);
        expect(plot.height).toBeGreaterThan(initialHeight);
    });
    
    test('tooltip shows correct values', () => {
        const testData = {
            samples: [0, 0, 0, 1, 1],  // 60% state 0, 40% state 1
            title: 'Tooltip Test'
        };
        
        const plot = new DistributionPlot(container, testData);
        plot.render();
        
        // Create tooltip if it doesn't exist
        if (!d3.select('.plot-tooltip').size()) {
            d3.select('body').append('div')
                .attr('class', 'plot-tooltip')
                .style('opacity', 0);
        }
        
        // Simulate mouseover on first bar
        const firstBar = container.select('.bar');
        const event = new MouseEvent('mouseover', {
            bubbles: true,
            cancelable: true,
            view: window
        });
        firstBar.node().dispatchEvent(event);
        
        // Check tooltip content
        const tooltip = d3.select('.plot-tooltip');
        expect(tooltip.empty()).toBeFalsy();
        expect(tooltip.style('opacity')).not.toBe('0');
        expect(tooltip.html()).toContain('60%');  // Should show correct percentage
        
        // Simulate mouseout
        const mouseoutEvent = new MouseEvent('mouseout', {
            bubbles: true,
            cancelable: true,
            view: window
        });
        firstBar.node().dispatchEvent(mouseoutEvent);
        
        // Tooltip should be hidden
        expect(tooltip.style('opacity')).toBe('0');
    });
}); 