"""
Simplified linear elastic solver for quick structural analysis.

This module implements a basic finite element solver for tetrahedral elements
using linear elastic theory. It's optimized for speed over accuracy for rapid
feedback during design iterations.
"""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, List
from loguru import logger


def compute_element_stiffness(
    nodes: np.ndarray, young: float, poisson: float
) -> np.ndarray:
    """
    Compute the stiffness matrix for a linear tetrahedral element.

    Args:
        nodes: 4x3 array of node coordinates
        young: Young's modulus
        poisson: Poisson's ratio

    Returns:
        12x12 element stiffness matrix
    """
    # Calculate element volume and shape function derivatives
    # For a tetrahedral element, the Jacobian is the matrix of node differences
    # J = [x2-x1  y2-y1  z2-z1]
    #     [x3-x1  y3-y1  z3-z1]
    #     [x4-x1  y4-y1  z4-z1]
    J = nodes[1:] - nodes[0]  # This gives us a 3x3 matrix
    volume = abs(np.linalg.det(J)) / 6.0

    # B matrix (strain-displacement)
    B = np.linalg.inv(J).T
    B_matrix = np.zeros((6, 12))

    # Fill B matrix for each node
    for i in range(4):
        B_matrix[0, 3 * i] = B[0, 0]
        B_matrix[1, 3 * i + 1] = B[1, 1]
        B_matrix[2, 3 * i + 2] = B[2, 2]
        B_matrix[3, 3 * i : 3 * i + 2] = [B[1, 0], B[0, 1]]
        B_matrix[4, 3 * i + 1 : 3 * i + 3] = [B[2, 1], B[1, 2]]
        B_matrix[5, [3 * i, 3 * i + 2]] = [B[2, 0], B[0, 2]]

    # D matrix (constitutive)
    factor = young / ((1 + poisson) * (1 - 2 * poisson))
    D = factor * np.array(
        [
            [1 - poisson, poisson, poisson, 0, 0, 0],
            [poisson, 1 - poisson, poisson, 0, 0, 0],
            [poisson, poisson, 1 - poisson, 0, 0, 0],
            [0, 0, 0, (1 - 2 * poisson) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * poisson) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * poisson) / 2],
        ]
    )

    return volume * B_matrix.T @ D @ B_matrix


def assemble_system(
    nodes: np.ndarray, elements: np.ndarray, material: Dict
) -> Tuple[csc_matrix, int]:
    """
    Assemble the global stiffness matrix.

    Args:
        nodes: Nx3 array of node coordinates
        elements: Ex4 array of element connectivity
        material: Dictionary containing material properties

    Returns:
        Tuple of (sparse stiffness matrix, number of DOFs)
    """
    n_nodes = len(nodes)
    n_elements = len(elements)
    ndof = 3 * n_nodes

    # Pre-allocate COO matrix arrays
    n_entries = 144 * n_elements  # 12x12 element matrix
    rows = np.zeros(n_entries, dtype=np.int32)
    cols = np.zeros(n_entries, dtype=np.int32)
    data = np.zeros(n_entries)

    idx = 0
    for el in range(n_elements):
        el_nodes = elements[el]
        node_coords = nodes[el_nodes]

        # Compute element stiffness
        K_el = compute_element_stiffness(
            node_coords,
            material["youngs_modulus_gpa"] * 1e9,  # Convert to Pa
            material["poissons_ratio"],
        )

        # Get DOF indices for this element
        dofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in el_nodes]).flatten()

        # Add to COO arrays
        for i in range(12):
            for j in range(12):
                rows[idx] = dofs[i]
                cols[idx] = dofs[j]
                data[idx] = K_el[i, j]
                idx += 1

    # Create sparse matrix
    K = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsc()
    return K, ndof


def solve_static(
    K: csc_matrix, ndof: int, load_vector: np.ndarray, fixed_dofs: List[int]
) -> np.ndarray:
    """
    Solve the static problem Ku = f with constraints.

    Args:
        K: Global stiffness matrix
        ndof: Number of degrees of freedom
        load_vector: Global force vector
        fixed_dofs: List of constrained DOFs

    Returns:
        Displacement vector
    """
    # Remove constrained DOFs
    free_dofs = list(set(range(ndof)) - set(fixed_dofs))
    K_free = K[free_dofs, :][:, free_dofs]
    f_free = load_vector[free_dofs]

    # Solve system
    u_free = spsolve(K_free, f_free)

    # Reconstruct full solution
    u = np.zeros(ndof)
    u[free_dofs] = u_free
    return u


def compute_stresses(
    nodes: np.ndarray, elements: np.ndarray, displacements: np.ndarray, material: Dict
) -> np.ndarray:
    """
    Compute von Mises stresses for each element.

    Args:
        nodes: Node coordinates
        elements: Element connectivity
        displacements: Nodal displacement vector
        material: Material properties

    Returns:
        Array of von Mises stresses for each element in Pa
    """
    try:
        n_elements = len(elements)
        von_mises = np.zeros(n_elements)

        young = material["youngs_modulus_gpa"] * 1e9  # Convert to Pa
        poisson = material["poissons_ratio"]

        # D matrix (constitutive)
        factor = young / ((1 + poisson) * (1 - 2 * poisson))
        D = factor * np.array(
            [
                [1 - poisson, poisson, poisson, 0, 0, 0],
                [poisson, 1 - poisson, poisson, 0, 0, 0],
                [poisson, poisson, 1 - poisson, 0, 0, 0],
                [0, 0, 0, (1 - 2 * poisson) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * poisson) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * poisson) / 2],
            ]
        )

        logger.debug(f"Processing {n_elements} elements")
        logger.debug(f"Element array type: {elements.dtype}")
        logger.debug(f"Node array type: {nodes.dtype}")
        logger.debug(f"Displacement array type: {displacements.dtype}")

        for el in range(n_elements):
            try:
                el_nodes = elements[el]
                node_coords = nodes[el_nodes]
                # Convert indices to integers and flatten
                node_indices = np.array(
                    [[3 * int(n), 3 * int(n) + 1, 3 * int(n) + 2] for n in el_nodes],
                    dtype=np.int64,
                ).flatten()
                el_disps = displacements[node_indices]

                # Calculate B matrix
                J = node_coords[1:] - node_coords[0]  # 3x3 Jacobian matrix
                B = np.linalg.inv(J).T
                B_matrix = np.zeros((6, 12))

                for i in range(4):
                    B_matrix[0, 3 * i] = B[0, 0]
                    B_matrix[1, 3 * i + 1] = B[1, 1]
                    B_matrix[2, 3 * i + 2] = B[2, 2]
                    B_matrix[3, 3 * i : 3 * i + 2] = [B[1, 0], B[0, 1]]
                    B_matrix[4, 3 * i + 1 : 3 * i + 3] = [B[2, 1], B[1, 2]]
                    B_matrix[5, [3 * i, 3 * i + 2]] = [B[2, 0], B[0, 2]]

                # Calculate stresses
                stress = D @ B_matrix @ el_disps

                # Calculate von Mises stress
                s11, s22, s33, s12, s23, s31 = stress
                von_mises[el] = np.sqrt(
                    0.5
                    * (
                        (s11 - s22) ** 2
                        + (s22 - s33) ** 2
                        + (s33 - s11) ** 2
                        + 6 * (s12**2 + s23**2 + s31**2)
                    )
                )

            except Exception as e:
                logger.error(f"Error processing element {el}: {str(e)}")
                logger.error(f"Element nodes: {el_nodes}")
                logger.error(f"Node coordinates shape: {node_coords.shape}")
                raise

        return von_mises  # Return in Pa

    except Exception as e:
        logger.error(f"Error in compute_stresses: {str(e)}")
        raise
