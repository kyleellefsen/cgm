export class PanelManager {
    constructor(onGraphResized) {
        this.onGraphResized = onGraphResized;
        this.setupResizers();
    }
    setupResizers() {
        this.setupVerticalResizer();
        const horizontalResizers = document.getElementsByClassName('horizontal-resizer');
        for (const resizer of horizontalResizers) {
            this.setupHorizontalResizer(resizer);
        }
    }
    setupVerticalResizer() {
        const resizer = document.getElementsByClassName('vertical-resizer')[0];
        const prevSibling = resizer.previousElementSibling;
        const nextSibling = resizer.nextElementSibling;
        let isResizing = false;
        let startX;
        let prevSiblingWidth;
        let nextSiblingWidth;
        const startResize = (e) => {
            isResizing = true;
            resizer?.classList.add('resizing');
            startX = e.pageX;
            prevSiblingWidth = prevSibling?.offsetWidth || 0;
            nextSiblingWidth = nextSibling?.offsetWidth || 0;
            document.documentElement.style.cursor = 'col-resize';
            e.preventDefault();
            e.stopPropagation();
        };
        const resize = (e) => {
            if (!isResizing)
                return;
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
        const stopResize = (e) => {
            if (!isResizing)
                return;
            isResizing = false;
            resizer?.classList.remove('resizing');
            document.documentElement.style.cursor = '';
            e.preventDefault();
            e.stopPropagation();
        };
        resizer?.addEventListener('mousedown', (e) => startResize(e));
        document.addEventListener('mousemove', (e) => resize(e));
        document.addEventListener('mouseup', (e) => stopResize(e));
        // Add window resize handler
        window.addEventListener('resize', () => {
            if (prevSibling && nextSibling) {
                // Get current flex basis values
                const prevBasis = parseFloat(prevSibling.style.flexBasis) || prevSibling.offsetWidth;
                const nextBasis = parseFloat(nextSibling.style.flexBasis) || nextSibling.offsetWidth;
                const totalWidth = prevBasis + nextBasis;
                // Calculate and preserve the ratio
                const prevRatio = (prevBasis / totalWidth * 100).toFixed(0);
                const nextRatio = (nextBasis / totalWidth * 100).toFixed(0);
                // Apply the preserved ratios
                prevSibling.style.flex = `1 1 ${prevRatio}%`;
                nextSibling.style.flex = `1 1 ${nextRatio}%`;
                prevSibling.style.width = '';
                nextSibling.style.width = '';
            }
            // Notify graph visualization of width change
            this.onGraphResized(prevSibling.offsetWidth);
        });
    }
    setupHorizontalResizer(resizer) {
        const prevPanel = resizer.previousElementSibling;
        const nextPanel = resizer.nextElementSibling;
        if (!prevPanel || !nextPanel)
            return;
        let isResizing = false;
        let startY;
        let startUpperHeight;
        let startLowerHeight;
        const startResize = (e) => {
            isResizing = true;
            resizer.classList.add('resizing');
            startY = e.pageY;
            startUpperHeight = prevPanel.offsetHeight;
            startLowerHeight = nextPanel.offsetHeight;
            document.documentElement.style.cursor = 'row-resize';
            e.preventDefault();
            e.stopPropagation();
        };
        const resize = (e) => {
            if (!isResizing)
                return;
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
//# sourceMappingURL=panel-manager.js.map