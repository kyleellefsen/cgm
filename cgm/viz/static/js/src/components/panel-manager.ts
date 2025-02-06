import { PanelConfig, PanelLayout } from '../types';

export class PanelManager {
    private static readonly GOLDEN_LAYOUT_CLASSES = {
        root: 'lm_goldenlayout',
        row: 'lm_row',
        column: 'lm_column',
        stack: 'lm_stack',
        component: 'lm_item',
        header: 'lm_header',
        content: 'lm_content'
    };

    private components: Map<string, (container: HTMLElement, state: any) => void> = new Map();
    private container: HTMLElement;

    constructor(containerId: string, panels: PanelConfig[]) {
        const container = document.getElementById(containerId);
        if (!container) {
            throw new Error(`Container with id '${containerId}' not found`);
        }
        
        this.container = container;
        // Add GoldenLayout root class
        container.className = `${container.className} ${PanelManager.GOLDEN_LAYOUT_CLASSES.root}`;
        
        // Initialize with default layout if none provided
        this.buildLayout(container, {
            type: 'row',
            children: panels.map(panel => ({
                type: 'component',
                component: panel
            }))
        });
        
        // Add GoldenLayout-style event listeners
        window.addEventListener('resize', () => this.updateSize());
    }

    public registerComponent(type: string, factory: (container: HTMLElement, state: any) => void): void {
        this.components.set(type, factory);
    }

    public createComponent(config: PanelConfig): HTMLElement {
        const component = document.createElement('div');
        component.className = `${PanelManager.GOLDEN_LAYOUT_CLASSES.component} ${config.type}-panel`;
        
        const factory = this.components.get(config.type);
        if (factory) {
            factory(component, config.componentState || {});
        } else {
            console.warn(`No component registered for type: ${config.type}`);
        }
        
        return component;
    }

    private buildLayout(parent: HTMLElement, config: PanelLayout): void {
        const element = document.createElement('div');
        
        // Add GoldenLayout-style classes
        switch(config.type) {
            case 'row':
                element.className = PanelManager.GOLDEN_LAYOUT_CLASSES.row;
                element.style.display = 'flex';
                element.style.flexDirection = 'row';
                break;
                
            case 'column':
                element.className = PanelManager.GOLDEN_LAYOUT_CLASSES.column;
                element.style.display = 'flex';
                element.style.flexDirection = 'column';
                break;
                
            case 'stack':
                element.className = PanelManager.GOLDEN_LAYOUT_CLASSES.stack;
                break;
                
            case 'component':
                if (config.component) {
                    const component = this.createComponent(config.component);
                    element.appendChild(component);
                }
                return;
        }

        // Handle children recursively
        config.children?.forEach(childConfig => {
            this.buildLayout(element, childConfig);
            
            // Add GoldenLayout-style splitters between siblings
            if (config.type === 'row' || config.type === 'column') {
                const splitter = document.createElement('div');
                splitter.className = 'lm_splitter';
                splitter.dataset.orientation = config.type === 'row' ? 'vertical' : 'horizontal';
                element.appendChild(splitter);
            }
        });

        parent.appendChild(element);
    }

    public addComponent(config: PanelConfig, parent?: HTMLElement): void {
        const component = this.createComponent(config);
        (parent || this.container).appendChild(component);
    }

    public removeComponent(component: HTMLElement): void {
        component.remove();
    }

    public updateSize(): void {
        // GoldenLayout-style size update propagation
        this.container.querySelectorAll(`.${PanelManager.GOLDEN_LAYOUT_CLASSES.component}`).forEach(component => {
            const width = component.parentElement?.offsetWidth || 0;
            const height = component.parentElement?.offsetHeight || 0;
            component.dispatchEvent(new CustomEvent('resize', { 
                detail: { width, height }
            }));
        });
    }

    public removeAllComponents(): void {
        if (this.container) this.container.innerHTML = '';
    }
} 