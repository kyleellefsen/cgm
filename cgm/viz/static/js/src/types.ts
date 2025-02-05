import * as d3 from 'd3';

// Node types
export interface Node {
    id: string;
    states: number;
    type?: string;
    conditioned_state: number;
    cpd?: string;
    x?: number;
    y?: number;
    fx?: number | null;
    fy?: number | null;
    vx?: number;
    vy?: number;
    width?: number;
    height?: number;
    isDragging?: boolean;
    isPinned?: boolean;
    evidence?: number;
}

export interface SimulationNode extends Node {
    index?: number;
}

export interface Link {
    source: string | Node | SimulationNode;
    target: string | Node | SimulationNode;
}

export interface SimulationLink extends d3.SimulationLinkDatum<SimulationNode> {
    source: SimulationNode;
    target: SimulationNode;
    id?: string;
}

export interface GraphData {
    nodes: Node[];
    links: Link[];
}

export interface GraphState {
    conditions: Record<string, number>;
    lastSamplingResult: SamplingResult | null;
}

export interface PlotData {
    variable?: string;
    title?: string;
    x_values?: number[];
    y_values?: number[];
}

export interface SamplingSettings {
    method: string;
    sampleSize: number;
    autoUpdate: boolean;
    burnIn: number;
    thinning: number;
    randomSeed: number | null;
    cacheResults: boolean;
}

export interface SamplingResult {
    totalSamples: number;
    acceptedSamples: number;
    rejectedSamples: number;
}

export interface SamplingMetadata {
    seed: number | null;
    timestamp: number | null;
    num_samples: number;
    conditions: Record<string, number>;
}

// D3 related types
export type D3Selection<T extends Element = HTMLElement> = d3.Selection<T, unknown, HTMLElement, unknown>;
export type D3DivSelection = D3Selection<HTMLDivElement>;
export type D3SVGSelection = D3Selection<SVGElement>;
export type D3SVGGSelection = D3Selection<SVGGElement>;
export type D3SVGTextSelection = D3Selection<SVGTextElement>;
export type D3SVGRectSelection = D3Selection<SVGRectElement>;
export type D3Transition<T extends Element = HTMLElement> = d3.Transition<T, unknown, HTMLElement, unknown>;

// D3 Simulation types
export type ForceSimulation = d3.Simulation<SimulationNode, SimulationLink>;
export type NodeSelection = d3.Selection<SVGGElement, SimulationNode, SVGGElement, unknown>;
export type LinkSelection = d3.Selection<SVGLineElement, SimulationLink, SVGGElement, unknown>;

// Extend D3Selection interface with commonly used methods
declare module 'd3' {
    interface Selection<GElement extends d3.BaseType, Datum, PElement extends d3.BaseType, PDatum> {
        transition(): d3.Transition<GElement, Datum, PElement, PDatum>;
        style(name: string, value: string | number | ((d: Datum) => string | number)): this;
        attr(name: string, value: string | number | ((d: Datum) => string | number)): this;
        html(value?: string): this;
        text(value: string | ((d: Datum) => string)): this;
        call(fn: (selection: this) => void): this;
        empty(): boolean;
        node(): Element | null;
    }
}

// Add new types for node distribution API
export interface NodeDistributionError {
    error_type: string;
    message: string;
    details?: string;
}

export interface NodeDistributionSuccess {
    node_name: string;
    codomain: "counts" | "normalized_counts";
    x_values: number[];
    y_values: number[];
}

export interface NodeDistributionResponse {
    success: boolean;
    error?: NodeDistributionError;
    result?: NodeDistributionSuccess;
}

// Panel Management Types
export interface PanelConfig {
    id: string;                    // Unique identifier for the panel
    title?: string;               // Panel header title
    minWidth?: number;           // Minimum width constraint
    minHeight?: number;          // Minimum height constraint
    collapsible?: boolean;       // Whether panel can be collapsed
    onResize?: (width: number, height: number) => void;
    onCollapse?: (collapsed: boolean) => void;
    initialContent?: HTMLElement | string;
}

export interface PanelDimensions {
    width: number;
    height: number;
}

export interface ResizeEvent {
    panelId: string;
    dimensions: PanelDimensions;
}

export type ResizeCallback = (event: ResizeEvent) => void;
export type CollapseCallback = (panelId: string, collapsed: boolean) => void; 