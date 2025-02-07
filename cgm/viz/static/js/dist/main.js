// Main entry point for the visualization
import { GraphVisualization } from '/components/graph-visualization';
export { GraphVisualization } from '/components/graph-visualization';
export { PlotManager } from '/components/plot-manager';
export { SamplingControls } from '/components/sampling-controls';
export * from '/types';
// Initialize visualization
let initialized = false;
async function initializeViz() {
    if (!initialized) {
        try {
            window.graphViz = new GraphVisualization();
            initialized = true;
        }
        catch (error) {
            console.error('Failed to initialize visualization:', error);
        }
    }
}
// Initialize immediately since we're using modules
initializeViz();
// Global error handler
window.addEventListener('error', event => console.error('Script Error:', event.error));
//# sourceMappingURL=main.js.map