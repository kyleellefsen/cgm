interface HTMLElementWithStyle extends HTMLElement {
    style: CSSStyleDeclaration;
    offsetWidth: number;
    offsetHeight: number;
}

export class PanelManager {
    private onGraphResized: (newWidth: number) => void;

    constructor(onGraphResized: (newWidth: number) => void) {
        this.onGraphResized = onGraphResized;
        this.setupResizers();
    }

    setupResizers(): void {
        this.setupVerticalResizer();
        const horizontalResizers = document.getElementsByClassName('horizontal-resizer');
        for (const resizer of horizontalResizers) {
            this.setupHorizontalResizer(resizer as HTMLElement);
        }
    }

    private setupVerticalResizer() {
        const resizer = document.getElementsByClassName('vertical-resizer')[0];
        const prevSibling = resizer.previousElementSibling as HTMLElementWithStyle;
        const nextSibling = resizer.nextElementSibling as HTMLElementWithStyle;
        
        let isResizing = false;
        let startX: number;
        let prevSiblingWidth: number;
        let nextSiblingWidth: number;
        
        const startResize = (e: MouseEvent) => {
            isResizing = true;
            resizer?.classList.add('resizing');
            startX = e.pageX;
            prevSiblingWidth = prevSibling?.offsetWidth || 0;
            nextSiblingWidth = nextSibling?.offsetWidth || 0;
            document.documentElement.style.cursor = 'col-resize';
            e.preventDefault();
            e.stopPropagation();
        };
        
        const resize = (e: MouseEvent) => {
            if (!isResizing) return;
            
            const dx = e.pageX - startX;
            
            // Calculate new widths
            const newPrevSiblingWidth = prevSiblingWidth + dx;
            const newNextSiblingWidth = nextSiblingWidth - dx;
            
            // Apply minimum widths
            if (newPrevSiblingWidth >= 200 && newNextSiblingWidth >= 200) {
                // Use flex-basis instead of fixed widths to maintain flex behavior
                prevSibling.style.flexBasis = `${newPrevSiblingWidth}px`;
                nextSibling.style.flexBasis = `${newNextSiblingWidth}px`;
                
                // Notify graph visualization of width change
                this.onGraphResized(newPrevSiblingWidth);
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
        
        resizer?.addEventListener('mousedown', (e: Event) => startResize(e as MouseEvent));
        document.addEventListener('mousemove', (e: Event) => resize(e as MouseEvent));
        document.addEventListener('mouseup', (e: Event) => stopResize(e as MouseEvent));
        
        // Add window resize handler
        window.addEventListener('resize', () => {
            if (prevSibling && nextSibling) {
                // Reset flex layout
                prevSibling.style.flex = '1 1 70%';
                nextSibling.style.flex = '1 1 30%';
                prevSibling.style.width = '';
                nextSibling.style.width = '';
            }
            
            // Notify graph visualization of width change
            this.onGraphResized(prevSibling.offsetWidth);
        });
    }

    private setupHorizontalResizer(resizer: HTMLElement): void {
        const prevPanel = resizer.previousElementSibling as HTMLElementWithStyle;
        const nextPanel = resizer.nextElementSibling as HTMLElementWithStyle;
        
        if (!prevPanel || !nextPanel) return;

        let isResizing = false;
        let startY: number;
        let startUpperHeight: number;
        let startLowerHeight: number;

        const startResize = (e: MouseEvent) => {
            isResizing = true;
            resizer.classList.add('resizing');
            startY = e.pageY;
            startUpperHeight = prevPanel.offsetHeight;
            startLowerHeight = nextPanel.offsetHeight;
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
                prevPanel.style.flex = 'none';
                nextPanel.style.flex = 'none';
                prevPanel.style.height = `${newUpperHeight}px`;
                nextPanel.style.height = `${newLowerHeight}px`;
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
