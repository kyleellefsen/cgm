import { PanelConfig } from '../types';

export class PanelManager {
    private container: HTMLElement;

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

        // Set minimum dimensions as CSS variables
        if (config.minWidth) {
            panel.style.setProperty('--min-width', `${config.minWidth}px`);
        }
        if (config.minHeight) {
            panel.style.setProperty('--min-height', `${config.minHeight}px`);
        }

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
        let startX: number = 0;
        let startY: number = 0;

        this.container.addEventListener('mousedown', (e: MouseEvent) => {
            const target = e.target as HTMLElement;
            if (target.classList.contains('resizer')) {
                activeResizer = target;
                startX = e.pageX;
                startY = e.pageY;
                document.body.classList.add('resizing');
            }
        });

        document.addEventListener('mousemove', (e: MouseEvent) => {
            if (!activeResizer) return;

            const isVertical = activeResizer.classList.contains('vertical');
            const dx = e.pageX - startX;
            const dy = e.pageY - startY;

            if (isVertical) {
                const prev = activeResizer.previousElementSibling as HTMLElement;
                const next = activeResizer.nextElementSibling as HTMLElement;
                if (prev && next) {
                    const prevWidth = prev.offsetWidth + dx;
                    const nextWidth = next.offsetWidth - dx;
                    const prevMin = parseFloat(getComputedStyle(prev).getPropertyValue('--min-width') || '0');
                    const nextMin = parseFloat(getComputedStyle(next).getPropertyValue('--min-width') || '0');

                    if (prevWidth >= prevMin && nextWidth >= nextMin) {
                        prev.style.width = `${prevWidth}px`;
                        next.style.width = `${nextWidth}px`;
                        startX = e.pageX;
                    }
                }
            } else {
                const prev = activeResizer.previousElementSibling as HTMLElement;
                const next = activeResizer.nextElementSibling as HTMLElement;
                if (prev && next) {
                    const prevHeight = prev.offsetHeight + dy;
                    const nextHeight = next.offsetHeight - dy;
                    const prevMin = parseFloat(getComputedStyle(prev).getPropertyValue('--min-height') || '0');
                    const nextMin = parseFloat(getComputedStyle(next).getPropertyValue('--min-height') || '0');

                    if (prevHeight >= prevMin && nextHeight >= nextMin) {
                        prev.style.height = `${prevHeight}px`;
                        next.style.height = `${nextHeight}px`;
                        startY = e.pageY;
                    }
                }
            }
        });

        document.addEventListener('mouseup', () => {
            if (activeResizer) {
                activeResizer = null;
                document.body.classList.remove('resizing');
            }
        });
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