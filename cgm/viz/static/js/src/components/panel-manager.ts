import { 
    PanelConfig, 
    PanelDimensions, 
    ResizeEvent, 
    ResizeCallback, 
    CollapseCallback 
} from '../types';

export class PanelManager {
    private panels: Map<string, PanelConfig>;
    private container: HTMLElement;
    private resizeObserver: ResizeObserver;
    private resizeCallbacks: Set<ResizeCallback>;
    private collapseCallbacks: Set<CollapseCallback>;

    constructor(containerId: string, panels: PanelConfig[]) {
        const container = document.getElementById(containerId);
        if (!container) {
            throw new Error(`Container with id '${containerId}' not found`);
        }
        this.container = container;
        this.panels = new Map(panels.map(p => [p.id, p]));
        this.resizeCallbacks = new Set();
        this.collapseCallbacks = new Set();
        
        this.initPanels();
        this.setupResizeObserver();
        this.setupVerticalResizer();
        this.setupHorizontalResizers();
    }

    private initPanels(): void {
        this.panels.forEach((config, id) => {
            const panel = document.querySelector(`[data-panel-id="${id}"]`) as HTMLElement;
            if (!panel) return;

            // Apply initial dimensions and constraints
            if (config.defaultWidth) {
                panel.style.width = `${config.defaultWidth}px`;
            }
            if (config.flex) {
                panel.style.flex = config.flex;
            }
            if (config.minWidth) {
                panel.style.minWidth = `${config.minWidth}px`;
            }
            if (config.minHeight) {
                panel.style.minHeight = `${config.minHeight}px`;
            }
            
            // Setup collapsible behavior if enabled
            if (config.collapsible) {
                const header = panel.querySelector('.panel-header');
                if (header) {
                    header.addEventListener('click', () => this.togglePanel(id));
                }
                if (config.defaultCollapsed) {
                    this.togglePanel(id);
                }
            }

            // Set initial content if provided
            if (config.initialContent) {
                this.setPanelContent(id, config.initialContent);
            }
        });
    }

    private setupVerticalResizer(): void {
        const resizer = document.getElementById('vertical-resizer');
        if (!resizer) return;

        let isResizing = false;
        let startX: number;
        let startWidths: Map<string, number> = new Map();

        const startResize = (e: MouseEvent) => {
            isResizing = true;
            startX = e.pageX;
            document.body.classList.add('panel-manager-resizing');
            
            // Store starting widths of affected panels
            this.panels.forEach((config, id) => {
                const panel = document.querySelector(`[data-panel-id="${id}"]`) as HTMLElement;
                if (panel) {
                    startWidths.set(id, panel.offsetWidth);
                }
            });

            e.preventDefault();
            e.stopPropagation();
        };

        const resize = (e: MouseEvent) => {
            if (!isResizing) return;
            
            const dx = e.pageX - startX;
            this.updatePanelWidths(dx, startWidths);

            e.preventDefault();
            e.stopPropagation();
        };

        const stopResize = (e: MouseEvent) => {
            if (!isResizing) return;
            isResizing = false;
            document.body.classList.remove('panel-manager-resizing');
            e.preventDefault();
            e.stopPropagation();
        };

        resizer.addEventListener('mousedown', startResize);
        document.addEventListener('mousemove', resize);
        document.addEventListener('mouseup', stopResize);
    }

    private setupHorizontalResizers(): void {
        const resizers = document.getElementsByClassName('horizontal-resizer');
        Array.from(resizers).forEach(resizer => {
            let isResizing = false;
            let startY: number;
            let upperPanel: HTMLElement;
            let lowerPanel: HTMLElement;
            let startHeights: { upper: number; lower: number };

            const startResize = (e: MouseEvent) => {
                upperPanel = resizer.previousElementSibling as HTMLElement;
                lowerPanel = resizer.nextElementSibling as HTMLElement;
                if (!upperPanel || !lowerPanel) return;

                isResizing = true;
                startY = e.pageY;
                startHeights = {
                    upper: upperPanel.offsetHeight,
                    lower: lowerPanel.offsetHeight
                };
                document.body.classList.add('panel-manager-resizing');
                e.preventDefault();
                e.stopPropagation();
            };

            const resize = (e: MouseEvent) => {
                if (!isResizing) return;
                
                const dy = e.pageY - startY;
                this.updatePanelHeights(upperPanel, lowerPanel, dy, startHeights);
                e.preventDefault();
                e.stopPropagation();
            };

            const stopResize = (e: MouseEvent) => {
                if (!isResizing) return;
                isResizing = false;
                document.body.classList.remove('panel-manager-resizing');
                e.preventDefault();
                e.stopPropagation();
            };

            resizer.addEventListener('mousedown', startResize);
            document.addEventListener('mousemove', resize);
            document.addEventListener('mouseup', stopResize);
        });
    }

    private setupResizeObserver(): void {
        this.resizeObserver = new ResizeObserver(entries => {
            entries.forEach(entry => {
                const panel = entry.target as HTMLElement;
                const panelId = panel.dataset.panelId;
                if (!panelId) return;

                const event: ResizeEvent = {
                    panelId,
                    dimensions: {
                        width: entry.contentRect.width,
                        height: entry.contentRect.height
                    }
                };

                // Call panel-specific resize handler if it exists
                const config = this.panels.get(panelId);
                config?.onResize?.(event.dimensions);

                // Call global resize handlers
                this.resizeCallbacks.forEach(callback => callback(event));
            });
        });

        // Observe all panels
        this.panels.forEach((_, id) => {
            const panel = document.querySelector(`[data-panel-id="${id}"]`);
            if (panel) {
                this.resizeObserver.observe(panel);
            }
        });
    }

    public onResize(callback: ResizeCallback): void {
        this.resizeCallbacks.add(callback);
    }

    public onCollapse(callback: CollapseCallback): void {
        this.collapseCallbacks.add(callback);
    }

    public togglePanel(panelId: string): void {
        const panel = document.querySelector(`[data-panel-id="${panelId}"]`);
        if (!panel) return;

        const isCollapsed = panel.classList.toggle('collapsed');
        
        // Call panel-specific collapse handler if it exists
        const config = this.panels.get(panelId);
        config?.onCollapse?.(isCollapsed);

        // Call global collapse handlers
        this.collapseCallbacks.forEach(callback => callback(panelId, isCollapsed));
    }

    public getPanel(id: string): HTMLElement | null {
        return document.querySelector(`[data-panel-id="${id}"]`);
    }

    public setPanelContent(id: string, content: HTMLElement | string): void {
        const panel = this.getPanel(id);
        if (!panel) return;
        
        const contentContainer = panel.querySelector('.panel-content');
        if (contentContainer) {
            if (content instanceof HTMLElement) {
                contentContainer.innerHTML = '';
                contentContainer.appendChild(content);
            } else {
                contentContainer.innerHTML = content;
            }
        }
    }

    public getPanelDimensions(panelId: string): PanelDimensions | null {
        const panel = this.getPanel(panelId);
        if (!panel) return null;

        return {
            width: panel.offsetWidth,
            height: panel.offsetHeight
        };
    }

    private updatePanelWidths(dx: number, startWidths: Map<string, number>): void {
        this.panels.forEach((config, id) => {
            const panel = document.querySelector(`[data-panel-id="${id}"]`) as HTMLElement;
            if (!panel) return;

            const startWidth = startWidths.get(id);
            if (startWidth === undefined) return;

            const newWidth = startWidth + dx;
            if (newWidth >= (config.minWidth || 200)) {
                panel.style.flex = 'none';
                panel.style.width = `${newWidth}px`;
            }
        });
    }

    private updatePanelHeights(
        upperPanel: HTMLElement, 
        lowerPanel: HTMLElement, 
        dy: number, 
        startHeights: { upper: number; lower: number }
    ): void {
        const upperPanelId = upperPanel.dataset.panelId;
        const lowerPanelId = lowerPanel.dataset.panelId;
        
        if (!upperPanelId || !lowerPanelId) return;

        const upperConfig = this.panels.get(upperPanelId);
        const lowerConfig = this.panels.get(lowerPanelId);

        const newUpperHeight = startHeights.upper + dy;
        const newLowerHeight = startHeights.lower - dy;

        if (newUpperHeight >= (upperConfig?.minHeight || 100) && 
            newLowerHeight >= (lowerConfig?.minHeight || 100)) {
            upperPanel.style.flex = 'none';
            lowerPanel.style.flex = 'none';
            upperPanel.style.height = `${newUpperHeight}px`;
            lowerPanel.style.height = `${newLowerHeight}px`;
        }
    }
} 