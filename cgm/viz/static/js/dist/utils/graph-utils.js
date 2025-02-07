/**
 * Calculate the width of a node based on its text content width and padding
 */
export function calculateNodeWidth(textWidth, padding) {
    return textWidth + (padding * 2);
}
/**
 * Calculate the angle between two points
 */
export function calculateAngle(source, target) {
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    return Math.atan2(dy, dx);
}
/**
 * Calculate the point where a line intersects with the edge of a node
 */
export function calculateNodeEdgePoint(node, angle, width, height) {
    return {
        x: node.x + Math.cos(angle) * (width / 2),
        y: node.y + Math.sin(angle) * (height / 2)
    };
}
/**
 * Calculate the endpoints for a link between two nodes
 */
export function calculateLinkEndpoints(source, target) {
    const angle = calculateAngle(source, target);
    const sourcePoint = calculateNodeEdgePoint(source, angle, source.width, source.height);
    const targetPoint = calculateNodeEdgePoint(target, angle + Math.PI, target.width, target.height);
    return {
        x1: sourcePoint.x,
        y1: sourcePoint.y,
        x2: targetPoint.x,
        y2: targetPoint.y
    };
}
/**
 * Calculate CSS selectors for highlighting table cells based on node state
 */
export function calculateTableHighlightSelectors(node, hasParents) {
    if (node.conditioned_state === -1)
        return '';
    return hasParents ?
        `td[data-variable="${node.id}"][data-value="${node.conditioned_state}"], 
         th[data-variable="${node.id}"][data-value="${node.conditioned_state}"]` :
        `td:nth-child(${node.conditioned_state + 1}), 
         th:nth-child(${node.conditioned_state + 1})`;
}
/**
 * Calculate CSS classes for a node based on its properties
 */
export function calculateNodeClasses(node) {
    const classes = ['node'];
    if (node.type)
        classes.push(node.type);
    if (node.conditioned_state >= 0)
        classes.push('conditioned');
    return classes;
}
/**
 * Calculate initial node position near the center with some randomness
 */
export function calculateInitialNodePosition(width, height, randomness = 100) {
    return {
        x: width / 2 + (Math.random() - 0.5) * randomness,
        y: height / 2 + (Math.random() - 0.5) * randomness
    };
}
//# sourceMappingURL=graph-utils.js.map