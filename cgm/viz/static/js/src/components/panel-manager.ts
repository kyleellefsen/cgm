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
        this.resizeObserver = new ResizeObserver(this.handleResize.bind(this));
        
        this.initPanels();
        this.setupResizeObserver();
        this.setupResizers();
    }

    private handleResize(entries: ResizeObserverEntry[]): void {
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

    private setupResizers(): void {
        // Single event listener for all resizers using event delegation
        this.container.addEventListener('mousedown', e => {
            const target = e.target as HTMLElement;
            if (target.id === 'vertical-resizer') {
                this.handleVerticalResize(e);
            }
            else if (target.classList.contains('horizontal-resizer')) {
                this.handleHorizontalResize(e, target);
            }
        });
    }

    private handleVerticalResize(e: MouseEvent): void {
        let isResizing = true;
        const startX = e.pageX;
        const startWidths = new Map<string, number>();
        
        // Store initial panel widths
        this.panels.forEach((config, id) => {
            const panel = this.getPanel(id);
            if (panel) startWidths.set(id, panel.offsetWidth);
        });

        const mouseMove = (e: MouseEvent) => {
            if (!isResizing) return;
            const dx = e.pageX - startX;
            this.updatePanelWidths(dx, startWidths);
        };

        const mouseUp = () => {
            isResizing = false;
            document.body.classList.remove('panel-manager-resizing');
            document.removeEventListener('mousemove', mouseMove);
            document.removeEventListener('mouseup', mouseUp);
        };

        document.body.classList.add('panel-manager-resizing');
        document.addEventListener('mousemove', mouseMove);
        document.addEventListener('mouseup', mouseUp);
    }

    private handleHorizontalResize(e: MouseEvent, resizer: HTMLElement): void {
        let isResizing = true;
        const startY = e.pageY;
        const upperPanel = resizer.previousElementSibling as HTMLElement;
        const lowerPanel = resizer.nextElementSibling as HTMLElement;
        
        if (!upperPanel || !lowerPanel) return;
        
        const startHeights = {
            upper: upperPanel.offsetHeight,
            lower: lowerPanel.offsetHeight
        };

        const mouseMove = (e: MouseEvent) => {
            if (!isResizing) return;
            const dy = e.pageY - startY;
            this.updatePanelHeights(upperPanel, lowerPanel, dy, startHeights);
        };

        const mouseUp = () => {
            isResizing = false;
            document.body.classList.remove('panel-manager-resizing');
            document.removeEventListener('mousemove', mouseMove);
            document.removeEventListener('mouseup', mouseUp);
        };

        document.body.classList.add('panel-manager-resizing');
        document.addEventListener('mousemove', mouseMove);
        document.addEventListener('mouseup', mouseUp);
    }

    private updatePanelWidths(dx: number, startWidths: Map<string, number>): void {
        this.panels.forEach((config, id) => {
            const panel = this.getPanel(id);
            const startWidth = startWidths.get(id);
            if (panel && startWidth !== undefined) {
                const newWidth = startWidth + dx * (config.resizeWeight || 0);
                if (newWidth >= (config.minWidth || 50)) {
                    panel.style.flex = 'none';
                    panel.style.width = `${newWidth}px`;
                }
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

        if (newUpperHeight >= (upperConfig?.minHeight || 50) && 
            newLowerHeight >= (lowerConfig?.minHeight || 50)) {
            upperPanel.style.flex = 'none';
            lowerPanel.style.flex = 'none';
            upperPanel.style.height = `${newUpperHeight}px`;
            lowerPanel.style.height = `${newLowerHeight}px`;
        }
    }

    public onResize(callback: ResizeCallback): void {
        this.resizeCallbacks.add(callback);
    }

    public onCollapse(callback: CollapseCallback): void {
        this.collapseCallbacks.add(callback);
    }

    public togglePanel(panelId: string): void {
        const panel = this.getPanel(panelId);
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
            contentContainer.innerHTML = '';  // Clear existing content
            if (content instanceof HTMLElement) {
                contentContainer.appendChild(content);
            } else {
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = content.trim();
                
                // Only append the content div, not the header
                const contentDiv = tempDiv.querySelector('.panel-content');
                if (contentDiv) {
                    contentContainer.innerHTML = contentDiv.innerHTML;
                } else {
                    contentContainer.innerHTML = content;
                }
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

    private setupResizeObserver(): void {
        // Observe all panels
        this.panels.forEach((_, id) => {
            const panel = this.getPanel(id);
            if (panel) {
                this.resizeObserver.observe(panel);
            }
        });
    }
} 