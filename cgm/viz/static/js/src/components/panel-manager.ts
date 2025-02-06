import { PanelConfig } from '../types';

export class PanelManager {
    private container: HTMLElement;
    private startX: number = 0;
    private startY: number = 0;

    constructor(containerId: string, panels: PanelConfig[]) {
        const container = document.getElementById(containerId);
        if (!container) {
            throw new Error(`Container with id '${containerId}' not found`);
        }
        this.container = container;
        
        // Initialize panels
        panels.forEach(config => this.initPanel(config));
        
        // Setup resize handling
        this.setupResizeHandling();
    }

    private initPanel(config: PanelConfig): void {
        const panel = document.querySelector(`[data-panel-id="${config.id}"]`) as HTMLElement;
        if (!panel) return;


        // Setup collapsible behavior
        if (config.collapsible) {
            const header = panel.querySelector('.panel-header');
            if (header) {
                header.addEventListener('click', () => {
                    const isCollapsed = panel.classList.toggle('collapsed');
                    config.onCollapse?.(isCollapsed);
                });
            }
        }

        // Set initial content
        if (config.initialContent) {
            const contentContainer = panel.querySelector('.panel-content');
            if (contentContainer) {
                if (config.initialContent instanceof HTMLElement) {
                    contentContainer.appendChild(config.initialContent);
                } else {
                    contentContainer.innerHTML = config.initialContent;
                }
            }
        }

        // Setup resize observer
        const resizeObserver = new ResizeObserver(entries => {
            const entry = entries[0];
            if (entry && config.onResize) {
                config.onResize(entry.contentRect.width, entry.contentRect.height);
            }
        });
        resizeObserver.observe(panel);
    }

    private setupResizeHandling(): void {
        let activeResizer: HTMLElement | null = null;

        this.container.addEventListener('mousedown', (e: MouseEvent) => {
            const target = e.target as HTMLElement;
            if (target.classList.contains('vertical-resizer') || target.classList.contains('horizontal-resizer')) {
                activeResizer = target;
                this.startX = e.pageX;
                this.startY = e.pageY;
                document.body.classList.add('resizing');
            }
        });

        document.addEventListener('mousemove', (e: MouseEvent) => {
            if (!activeResizer) return;

            const isVertical = activeResizer.classList.contains('vertical-resizer');
            const dx = e.pageX - this.startX;
            const dy = e.pageY - this.startY;

            if (isVertical) {
                this.handleVerticalResize(activeResizer, dx, e.pageX);
            } else {
                this.handleHorizontalResize(activeResizer, dy, e.pageY);
            }
        });

        document.addEventListener('mouseup', () => {
            if (activeResizer) {
                activeResizer = null;
                document.body.classList.remove('resizing');
            }
        });
    }

    private handleVerticalResize(resizer: HTMLElement, dx: number, currentX: number): void {
        const prev = resizer.previousElementSibling as HTMLElement;
        const next = resizer.nextElementSibling as HTMLElement;
        if (prev && next) {
            const prevWidth = prev.offsetWidth + dx;
            const nextWidth = next.offsetWidth - dx;
            const prevMin = parseFloat(getComputedStyle(prev).getPropertyValue('--min-width') || '0');
            const nextMin = parseFloat(getComputedStyle(next).getPropertyValue('--min-width') || '0');

            if (prevWidth >= prevMin && nextWidth >= nextMin) {
                prev.style.width = `${prevWidth}px`;
                next.style.width = `${nextWidth}px`;
                this.startX = currentX;
            }
        }
    }

    private handleHorizontalResize(resizer: HTMLElement, dy: number, currentY: number): void {
        const prev = resizer.previousElementSibling as HTMLElement;
        const next = resizer.nextElementSibling as HTMLElement;
        if (prev && next) {
            const prevHeight = prev.offsetHeight + dy;
            const nextHeight = next.offsetHeight - dy;
            const prevMin = parseFloat(getComputedStyle(prev).getPropertyValue('--min-height') || '0');
            const nextMin = parseFloat(getComputedStyle(next).getPropertyValue('--min-height') || '0');

            if (prevHeight >= prevMin && nextHeight >= nextMin) {
                prev.style.height = `${prevHeight}px`;
                next.style.height = `${nextHeight}px`;
                this.startY = currentY;
            }
        }
    }

    public setPanelContent(id: string, content: HTMLElement | string): void {
        const panel = document.querySelector(`[data-panel-id="${id}"]`);
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
} 