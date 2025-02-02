/**
 * @jest-environment jsdom
 */

import * as d3 from 'd3';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { jest } from '@jest/globals';
import { GraphVisualization, PlotManager, DistributionPlot } from '../../cgm/viz/static/js/viz-graph.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Mock fetch for API calls
globalThis.fetch = jest.fn();

// Mock SamplingControls class
globalThis.SamplingControls = class {
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