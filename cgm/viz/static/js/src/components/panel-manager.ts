interface HTMLElementWithStyle extends HTMLElement {
    style: CSSStyleDeclaration;
    offsetWidth: number;
    offsetHeight: number;
}

export class PanelManager {
    private graphContainer: HTMLElementWithStyle;
    private panelsContainer: HTMLElementWithStyle;
    private onGraphResized: (newWidth: number) => void;

    constructor(onGraphResized: (newWidth: number) => void) {
        this.onGraphResized = onGraphResized;
        this.graphContainer = document.querySelector('.graph-container') as HTMLElementWithStyle;
        this.panelsContainer = document.querySelector('.panels-container') as HTMLElementWithStyle;
        this.setupResizers();
    }

    setupResizers(): void {
        this.setupVerticalResizer();
        this.setupHorizontalResizer('upper-resizer', '.cpd-table-panel', '.distribution-plot-panel');
        this.setupHorizontalResizer('plot-resizer', '.distribution-plot-panel', '.sampling-controls-panel');
    }

    private setupVerticalResizer() {
        const resizer = document.getElementById('vertical-resizer');
        
        let isResizing = false;
        let startX: number;
        let startGraphWidth: number;
        let startPanelsWidth: number;
        
        const startResize = (e: MouseEvent) => {
            isResizing = true;
            resizer?.classList.add('resizing');
            startX = e.pageX;
            startGraphWidth = this.graphContainer?.offsetWidth || 0;
            startPanelsWidth = this.panelsContainer?.offsetWidth || 0;
            document.documentElement.style.cursor = 'col-resize';
            e.preventDefault();
            e.stopPropagation();
        };
        
        const resize = (e: MouseEvent) => {
            if (!isResizing || !this.graphContainer || !this.panelsContainer) return;
            
            const dx = e.pageX - startX;
            
            // Calculate new widths
            const newGraphWidth = startGraphWidth + dx;
            const newPanelsWidth = startPanelsWidth - dx;
            
            // Apply minimum widths
            if (newGraphWidth >= 200 && newPanelsWidth >= 200) {
                // Use flex-basis instead of fixed widths to maintain flex behavior
                this.graphContainer.style.flexBasis = `${newGraphWidth}px`;
                this.panelsContainer.style.flexBasis = `${newPanelsWidth}px`;
                
                // Notify graph visualization of width change
                this.onGraphResized(newGraphWidth);
            }
            e.preventDefault();
            e.stopPropagation();
        };
        
        const stopResize = (e: MouseEvent) => {
            if (!isResizing) return;
            isResizing = false;
            resizer?.classList.remove('resizing');
            document.documentElement.style.cursor = '';
            e.preventDefault();
            e.stopPropagation();
        };
        
        resizer?.addEventListener('mousedown', startResize);
        document.addEventListener('mousemove', resize);
        document.addEventListener('mouseup', stopResize);

        // Add window resize handler
        window.addEventListener('resize', () => {
            if (this.graphContainer && this.panelsContainer) {
                // Reset flex layout
                this.graphContainer.style.flex = '1 1 70%';
                this.panelsContainer.style.flex = '1 1 30%';
                this.graphContainer.style.width = '';
                this.panelsContainer.style.width = '';
            }
            
            // Notify graph visualization of width change
            this.onGraphResized(this.graphContainer?.offsetWidth || 0);
        });
    }

    private setupHorizontalResizer(resizerId: string, upperSelector: string, lowerSelector: string): void {
        const resizer = document.getElementById(resizerId);
        const upperPanel = document.querySelector(upperSelector) as HTMLElementWithStyle;
        const lowerPanel = document.querySelector(lowerSelector) as HTMLElementWithStyle;
        
        if (!resizer || !upperPanel || !lowerPanel) return;

        let isResizing = false;
        let startY: number;
        let startUpperHeight: number;
        let startLowerHeight: number;

        const startResize = (e: MouseEvent) => {
            isResizing = true;
            resizer.classList.add('resizing');
            startY = e.pageY;
            startUpperHeight = upperPanel.offsetHeight;
            startLowerHeight = lowerPanel.offsetHeight;
            document.documentElement.style.cursor = 'row-resize';
            e.preventDefault();
            e.stopPropagation();
        };

        const resize = (e: MouseEvent) => {
            if (!isResizing) return;

            const dy = e.pageY - startY;
            const newUpperHeight = startUpperHeight + dy;
            const newLowerHeight = startLowerHeight - dy;

            if (newUpperHeight >= 100 && newLowerHeight >= 100) {
                upperPanel.style.flex = 'none';
                lowerPanel.style.flex = 'none';
                upperPanel.style.height = `${newUpperHeight}px`;
                lowerPanel.style.height = `${newLowerHeight}px`;
            }

            e.preventDefault();
            e.stopPropagation();
        };

        const stopResize = () => {
            isResizing = false;
            resizer.classList.remove('resizing');
            document.documentElement.style.cursor = '';
        };

        resizer.addEventListener('mousedown', startResize);
        document.addEventListener('mousemove', resize);
        document.addEventListener('mouseup', stopResize);
    }
}
