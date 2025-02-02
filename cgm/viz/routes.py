"""FastAPI routes for the visualization module."""
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pathlib
import time

from .state import state_instance
from .models import SamplingRequest, SamplingResponse
from ..inference.approximate import sampling
from ..inference.approximate.sampling import ForwardSamplingCertificate

def setup_routes(app: FastAPI, static_dir: pathlib.Path) -> None:
    """Setup all routes for the application."""
    
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def home() -> FileResponse:
        """Serve the visualization page."""
        return FileResponse(static_dir / "viz-layout.html")

    @app.get("/state")
    async def get_state() -> Dict:
        """Return the current graph state as JSON."""
        current_graph = state_instance.graph
        if not current_graph:
            return {"nodes": [], "links": []}

        current_state = state_instance.state
        nodes = [{
            "id": node.name,
            "states": node.num_states,
            "type": "effect",
            "conditioned_state": -1,
            "cpd": node.cpd.table.html() if node.cpd else None,
            **({"conditioned_state": int(current_state.values[current_state.schema.var_to_idx[node.name]])}
               if current_state and current_state.mask[current_state.schema.var_to_idx[node.name]] else {})
        } for node in current_graph.nodes]

        links = [{"source": parent.name, "target": node.name}
                for node in current_graph.nodes
                for parent in node.parents]

        return {"nodes": nodes, "links": links}

    @app.post("/condition/{node_id}/{state_value}")
    async def condition_node(node_id: str, state_value: int):
        """Update node's conditioned state"""
        if not state_instance.graph:
            return {"status": "no graph loaded"}
        
        if state_instance.condition(node_id, None if state_value == -1 else state_value):
            return {"status": "updated"}
        return {"status": "failed"}

    @app.post("/api/sample", response_model=SamplingResponse)
    async def generate_samples(request: SamplingRequest) -> SamplingResponse:
        """Generate samples from the current graph state."""
        current_graph = state_instance.graph
        if not current_graph:
            raise HTTPException(status_code=400, detail="No graph loaded")
            
        try:
            # Set random seed and get the seed used
            seed_used = state_instance.set_seed(request.options.random_seed)
            
            # Start with existing graph state if it exists, otherwise create new state with request conditions
            current_state = state_instance.state
            if current_state:
                # If we have existing state, create new state with combined conditions
                combined_conditions = state_instance.conditioned_nodes.copy()
                combined_conditions.update(request.conditions)
                current_state = current_graph.condition(**combined_conditions)
            else:
                # If no existing state, create new state with just request conditions
                current_state = current_graph.condition(**request.conditions)
            
            # Create the sampling certificate
            try:
                cert = ForwardSamplingCertificate(current_state)
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=(
                        f"Cannot use forward sampling with these conditions: {str(e)}. "
                        "Forward sampling requires that if a node is conditioned, all of its ancestors "
                        "must also be conditioned. Consider using a different sampling method or "
                        "adjusting your conditions to satisfy this requirement."
                    )
                )
                
            # Generate samples
            try:
                sample_array, _ = sampling.get_n_samples(
                    current_graph,
                    state_instance._rng,
                    num_samples=request.num_samples,
                    state=current_state,
                    certificate=cert,
                    return_array=True
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Sampling failed: {str(e)}"
                )
            
            if not isinstance(sample_array, sampling.SampleArray):
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected sample format: {type(sample_array)}"
                )
            
            # Store samples and metadata
            metadata = {
                'seed': seed_used,
                'timestamp': time.time(),
                'num_samples': request.num_samples,
                'conditions': request.conditions
            }
            state_instance.store_samples(sample_array, metadata)
            
            # Process results for the target variable (if specified)
            try:
                target_var = request.target_variable
                if target_var is None:
                    target_var = current_graph.nodes[0].name
                
                samples = state_instance.get_node_samples(target_var)
                if samples is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown variable: {target_var}"
                    )
                
                result = SamplingResponse(
                    total_samples=len(samples),
                    accepted_samples=len(samples),
                    rejected_samples=0,
                    samples=samples,
                    seed_used=seed_used,
                    target_variable=target_var
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process sampling results: {str(e)}"
                )
                
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"Unexpected error during sampling: {str(e)}\n{traceback.format_exc()}"
            print(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)

    @app.get("/api/node_distribution/{node_name}")
    async def get_node_distribution(node_name: str) -> Dict[str, Any]:
        """Get the distribution for a specific node from cached samples."""
        if not state_instance.graph:
            raise HTTPException(status_code=400, detail="No graph loaded")
            
        if state_instance._current_samples is None:
            raise HTTPException(
                status_code=400, 
                detail="No samples available. Generate samples first."
            )
        
        samples = state_instance.get_node_samples(node_name)
        if samples is None:
            raise HTTPException(status_code=400, detail=f"Unknown node: {node_name}")
        
        return {
            "samples": samples,
            "metadata": state_instance._samples_metadata,
            "node": node_name
        } 