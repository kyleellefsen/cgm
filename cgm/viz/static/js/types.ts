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
    source: string | Node;
    target: string | Node;
}

export interface SimulationLink extends d3.SimulationLinkDatum<SimulationNode> {
    source: SimulationNode;
    target: SimulationNode;
}

export interface GraphData {
    nodes: Node[];
    links: Link[];
}

export interface GraphState {
    conditions: Record<string, number>;
    lastSamplingResult: any | null;
}

export interface PlotData {
    variable?: string;
    title?: string;
    samples?: number[];
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