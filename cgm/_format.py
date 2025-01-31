"""Helper functions for formatting Factor and CPD tables for human-readable output."""
from typing import TYPE_CHECKING
import itertools
if TYPE_CHECKING:
    from .core import Factor, CPD



def _int_to_superscript(n: int) -> str:
    """Convert an integer to its Unicode superscript representation."""
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }
    return ''.join(superscript_map[d] for d in str(n))



def _format_factor_table(factor: 'Factor', row=None, col=None, fixed=None) -> str:
    """Format a Factor's values as a human-readable 2D table.
    
    Args:
        factor: A Factor object containing scope and values
        row: Variable name or index for row dimension (defaults to second-to-last dim)
        col: Variable name or index for column dimension (defaults to last dim)
        fixed: Dict mapping var names/indices to states for other dimensions
               (defaults to state 0 for all other dims)
    
    Example Usage:
    ```python
    # Default view (property access)
    print(phi.table)
    
    # Custom view (method access)
    print(phi.table(row='A', col='B', fixed={'C': 1}))
    ```
    """
    scope = factor.scope
    values = factor.values
    
    if len(scope) <= 1:
        # Special case for 0D or 1D factors
        lines = []
        header = f"{factor} |"
        if len(scope) == 1:
            var = scope[0]
            for x in range(var.num_states):
                header += f" {var.name}{_int_to_superscript(x)}".rjust(6)
        lines.append(header)
        lines.append("─" * (len(header) + 10))
        
        row = "    |"
        if len(scope) == 0:
            row += f" {values:>6.3f}"
        else:
            for i in range(values.shape[0]):
                row += f" {values[i]:>6.3f}"
        lines.append(row)
        return "\n".join(lines)

    # Handle dimension selection
    def get_var_index(var_spec, default):
        if var_spec is None:
            return default
        if isinstance(var_spec, int):
            return var_spec
        if isinstance(var_spec, str):
            # Find index of variable with matching name
            for idx, var in enumerate(scope):
                if var.name == var_spec:
                    return idx
            raise ValueError(f"Variable {var_spec} not found in factor scope")
        raise ValueError(f"Invalid variable specification: {var_spec}")

    row_idx = get_var_index(row, len(scope) - 2)
    col_idx = get_var_index(col, len(scope) - 1)

    # Validate indices
    if row_idx < 0 or row_idx >= len(scope):
        raise ValueError(f"Row index {row_idx} out of bounds")
    if col_idx < 0 or col_idx >= len(scope):
        raise ValueError(f"Column index {col_idx} out of bounds")
    if row_idx == col_idx:
        raise ValueError("Row and column dimensions must be different")

    # Get the variables for rows and columns
    row_var = scope[row_idx]
    col_var = scope[col_idx]

    # Process fixed dimensions
    fixed_dict = {}
    if fixed is not None:
        for var_spec, state in fixed.items():
            idx = get_var_index(var_spec, None)
            if idx in (row_idx, col_idx):
                raise ValueError(f"Cannot fix dimension used for rows/columns: {var_spec}")
            fixed_dict[idx] = state

    # For all other dimensions not specified, use state 0
    other_dims = [i for i in range(len(scope)) 
                  if i not in (row_idx, col_idx)]
    for dim in other_dims:
        if dim not in fixed_dict:
            fixed_dict[dim] = 0

    # Calculate widths for formatting
    PROB_WIDTH = 6  # Width for probability values
    COLUMN_WIDTH = 7  # Total column width including spacing
    row_label_width = max(len(str(row_var.name)) + 2, 6)  # Min width of 6

    lines = []

    # Create factor name showing fixed values
    if fixed_dict:
        fixed_str = ", ".join(f"{scope[dim].name}{_int_to_superscript(state)}"
                            for dim, state in sorted(fixed_dict.items()))
        factor_name = f"ϕ({fixed_str}, {row_var.name}, {col_var.name})"
    else:
        factor_name = str(factor)

    # Add table header with column variable states
    header = f"{factor_name} |"
    for x in range(col_var.num_states):
        header += f" {col_var.name}{_int_to_superscript(x)}".rjust(PROB_WIDTH)
    lines.append(header)

    # Add separator line
    total_width = row_label_width + 3 + (col_var.num_states * COLUMN_WIDTH)
    lines.append("─" * total_width)

    # Add data rows
    for row_state in range(row_var.num_states):
        row_label = f"{row_var.name}{_int_to_superscript(row_state)}"
        row = f"{row_label:{row_label_width}} |"

        # Construct index tuple for each cell
        for col_state in range(col_var.num_states):
            # Build the full index tuple
            idx = [0] * len(scope)
            # Fill in fixed states
            for dim, state in fixed_dict.items():
                idx[dim] = state
            # Fill in row and column states
            idx[row_idx] = row_state
            idx[col_idx] = col_state

            value = values[tuple(idx)]
            row += f" {value:>6.3f}"

        lines.append(row)

    return "\n".join(lines)



def _format_cpd_table(cpd: 'CPD') -> str:
    """Format a CPD's values as a human-readable table following Koller & Friedman notation.
    
    The format follows Figure 3.4 in "Probabilistic Graphical Models":
    - Child variable states are columns
    - Parent variable configurations are rows with superscript state indicators
    - Probabilities are formatted to 3 decimal places
    
    Args:
        cpd: A CPD object containing child, parents, and values attributes

    Example Usage:

    ```python
    g = cgm.Graph()
    A = g.node('A', 2)
    B = g.node('B', 3)
    C = g.node('C', 4)
    phi3 = g.P(C | [B, A])
    print(phi3.table)
    ```
    
    𝑃(C | B, A)  |    C⁰    C¹    C²    C³
    ───────────────────────────────────────────
    A⁰, B⁰       |  0.210  0.114  0.337  0.339
    A⁰, B¹       |  0.403  0.162  0.231  0.204
    A⁰, B²       |  0.093  0.065  0.596  0.245
    A¹, B⁰       |  0.742  0.150  0.063  0.044
    A¹, B¹       |  0.059  0.076  0.438  0.427
    A¹, B²       |  0.146  0.205  0.154  0.494

    """
    child = cpd.child
    parents = sorted(list(cpd.parents))

    # Find child dimension in the values array
    child_dim = cpd.scope.index(child)

    # Constants for formatting
    PROB_WIDTH = 6  # Width for probability values
    COLUMN_WIDTH = 7  # Total column width including spacing

    if not parents:
        # Case: No parents - P(x)
        prob_width = child.num_states * COLUMN_WIDTH
        total_width = max(prob_width + 4, 12)  # Min width of 12

        lines = []
        # Header with distribution name and child states
        header = f"{cpd} |"
        for x in range(child.num_states):
            header += f" {child.name}{_int_to_superscript(x)}".rjust(PROB_WIDTH)
        lines.append(header)
        lines.append("─" * total_width)

        # Single row of probabilities
        row = "    |"
        selector = [0] * len(cpd.scope)
        for x in range(child.num_states):
            selector[child_dim] = x
            row += f" {cpd.values[tuple(selector)]:>6.3f}"
        lines.append(row)
        return "\n".join(lines)

    else:
        # General case with parents
        # Calculate max width needed for row labels
        header_str = str(cpd).rjust(PROB_WIDTH)

        # Generate all parent state combinations to find max width needed
        parent_states = [range(p.num_states) for p in parents]
        max_label_width = len(header_str)
        for parent_vals in itertools.product(*parent_states):
            row_label = ", ".join(f"{p.name}{_int_to_superscript(v)}"
                                for p, v in zip(parents, parent_vals))
            max_label_width = max(max_label_width, len(row_label))

        # Add a small buffer
        label_width = max_label_width + 1

        lines = []
        # Header row with distribution name and child states
        header = f"{header_str:{label_width}} |"
        for x in range(child.num_states):
            header += f" {child.name}{_int_to_superscript(x)}".rjust(PROB_WIDTH)
        lines.append(header)

        # Calculate total width for separator line
        total_width = label_width + 3 + (child.num_states * COLUMN_WIDTH)
        lines.append("─" * total_width)

        # Generate rows for each parent configuration
        for parent_vals in itertools.product(*parent_states):
            row_label = ", ".join(f"{p.name}{_int_to_superscript(v)}"
                                for p, v in zip(parents, parent_vals))
            row = f"{row_label:{label_width}} |"

            # Create a selector array that puts values in the right dimensions
            selector = [0] * len(cpd.scope)
            # Map parent values to the correct dimensions
            for parent, val in zip(parents, parent_vals):
                parent_dim = cpd.scope.index(parent)
                selector[parent_dim] = val

            # Add probabilities for this parent configuration
            for x in range(child.num_states):
                selector[child_dim] = x
                prob = cpd.values[tuple(selector)]
                row += f" {prob:>6.3f}"
            lines.append(row)

        return "\n".join(lines)


def _format_cpd_as_html(cpd: 'CPD') -> str:
    """Format a CPD as an HTML table.

    Args:
        cpd: A CPD object containing child, parents, and values attributes

    Returns:
        An HTML string representing the CPD table.
    """
    child = cpd.child
    parents = sorted(list(cpd.parents))
    
    html = '<table class="cpd-table">\n'
    html += "  <thead>\n    <tr>\n"
    
    # Add header cells with variable names
    for parent in parents:
        html += f'      <th data-variable="{parent.name}">{parent.name}</th>\n'
    for i in range(child.num_states):
        html += f'      <th data-variable="{child.name}">{child.name}<sup>{i}</sup></th>\n'
    html += "    </tr>\n  </thead>\n"
    
    html += "  <tbody>\n"
    for idx in itertools.product(*[range(p.num_states) for p in parents]):
        html += "    <tr>\n"
        # Add parent states with data attributes
        for i, parent in enumerate(parents):
            html += f'      <td data-variable="{parent.name}" data-value="{idx[i]}">{idx[i]}</td>\n'
        
        # Get probabilities for this configuration
        selector = [idx[parents.index(p)] if p in parents else slice(None) for p in cpd.scope]
        probs = cpd.values[tuple(selector)]
        
        # Add probability cells
        for j in range(child.num_states):
            html += f'      <td data-variable="{child.name}" data-value="{j}">{probs[j]:.3f}</td>\n'
        html += "    </tr>\n"
    html += "  </tbody>\n</table>"
    
    return html


class FactorTableView:
    """Helper class to provide both property and method access to table formatting.
    
    This allows both:
        print(phi.table)  # Default view
        print(phi.table(row='A', col='B', fixed={'C': 1}))  # Custom view
    """
    def __init__(self, factor: 'Factor'):
        self.factor = factor

    def __call__(self, row=None, col=None, fixed=None) -> str:
        """Method access for custom table views."""
        return _format_factor_table(self.factor, row, col, fixed)

    def __str__(self) -> str:
        """Property access for default table view."""
        return _format_factor_table(self.factor)

class CPDTableView(FactorTableView):
    """Helper class to provide both property and method access to CPD table formatting.

    """
    def __init__(self, cpd: 'CPD'):
        super().__init__(cpd)
        self.cpd = cpd  # Store as CPD type specifically

    def __str__(self) -> str:
        return _format_cpd_table(self.cpd)  # Use cpd instead of factor

    def html(self) -> str:
        return _format_cpd_as_html(self.cpd)
