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
    Compute the stiffness matrix for a linear tetrahedral element (12x12).
    Uses shape function gradient formulation.
    """
    # Construct matrix for shape function coefficients
    X = np.ones((4, 4))
    X[:, 1:] = nodes
    detJ = np.linalg.det(X)

    volume = abs(detJ) / 6.0
    if volume < 1e-12:
        raise ValueError("Degenerate tetrahedral element with near-zero volume")

    # Shape function gradients (b, c, d coefficients)
    C = np.linalg.inv(X)

    grads = C[1:, :]  # Each column is the gradient of one shape function

    # Construct B matrix
    B = np.zeros((6, 12))
    for i in range(4):
        bi, ci, di = grads[0, i], grads[1, i], grads[2, i]
        B[:, 3 * i : 3 * i + 3] = [
            [bi, 0, 0],
            [0, ci, 0],
            [0, 0, di],
            [ci, bi, 0],
            [0, di, ci],
            [di, 0, bi],
        ]

    # Constitutive matrix D (isotropic)
    E = young
    nu = poisson
    factor = E / ((1 + nu) * (1 - 2 * nu))
    D = factor * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ]
    )

    return volume * B.T @ D @ B


def assemble_system(
    nodes: np.ndarray, elements: np.ndarray, material: Dict
) -> Tuple[csc_matrix, int]:
    """
    Assemble global stiffness matrix for linear elastic analysis.

    Args:
        nodes: Nx3 array of node coordinates
        elements: Ex4 array of element connectivity (4-node tets)
        material: Dictionary of material properties

    Returns:
        K: Global stiffness matrix (sparse)
        ndof: Number of degrees of freedom
    """
    n_nodes = len(nodes)
    ndof = 3 * n_nodes  # 3 DOFs per node

    # Material properties
    E = material["youngs_modulus_gpa"] * 1e9  # Convert GPa to Pa
    nu = material["poissons_ratio"]

    # Elastic constants
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # Constitutive matrix
    C = np.array(
        [
            [lambda_ + 2 * mu, lambda_, lambda_, 0, 0, 0],
            [lambda_, lambda_ + 2 * mu, lambda_, 0, 0, 0],
            [lambda_, lambda_, lambda_ + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ]
    )

    # Initialize COO matrix entries
    i_list = []
    j_list = []
    v_list = []

    # Assemble element matrices
    for el in elements:
        el_nodes = nodes[el]

        try:
            B, vol = compute_B_matrix(el_nodes)
            Ke = vol * B.T @ C @ B

            # Add to global matrix
            for i in range(4):
                for j in range(4):
                    for d1 in range(3):
                        for d2 in range(3):
                            i_list.append(3 * el[i] + d1)
                            j_list.append(3 * el[j] + d2)
                            v_list.append(Ke[3 * i + d1, 3 * j + d2])

        except ValueError as e:
            logger.warning(f"Skipping degenerate element: {str(e)}")
            continue

    # Create sparse matrix
    K = coo_matrix((v_list, (i_list, j_list)), shape=(ndof, ndof)).tocsc()

    return K, ndof


def solve_static(
    K: csc_matrix, ndof: int, f: np.ndarray, fixed_dofs: List[int]
) -> np.ndarray:
    """
    Solve static equilibrium Ku = f with fixed DOFs.

    Args:
        K: Global stiffness matrix
        ndof: Number of degrees of freedom
        f: Force vector
        fixed_dofs: List of fixed DOF indices

    Returns:
        u: Displacement vector
    """
    # Get free DOFs
    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Extract system for free DOFs
    K_free = K[free_dofs][:, free_dofs]
    f_free = f[free_dofs]

    # Add small value to diagonal for stability
    eps = 1e-10 * K_free.diagonal().mean()
    K_free.setdiag(K_free.diagonal() + eps)

    try:
        # Solve system
        u_free = spsolve(K_free, f_free)

        # Check for NaN values
        if np.any(np.isnan(u_free)):
            logger.error("NaN values in solution - system may be ill-conditioned")
            raise ValueError("Solver produced NaN values")

        # Reconstruct full solution
        u = np.zeros(ndof)
        u[free_dofs] = u_free

        return u

    except Exception as e:
        logger.error(f"Solver failed: {str(e)}")
        raise RuntimeError("Failed to solve system") from e


def compute_B_matrix(nodes: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the strain-displacement (B) matrix and volume for a tetrahedral element.
    """
    X = np.ones((4, 4))
    X[:, 1:] = nodes

    try:
        detJ = np.linalg.det(X)
        volume = abs(detJ) / 6.0

        # Check for degenerate elements
        if volume < 1e-12:  # Increased threshold for better stability
            raise ValueError(f"Degenerate element detected (volume: {volume:.2e})")

        # Use more stable matrix inversion
        C = np.linalg.solve(X, np.eye(4))  # Instead of direct inverse
        grads = C[1:, :]  # Gradient of shape functions

        B = np.zeros((6, 12))
        for i in range(4):
            bi, ci, di = grads[0, i], grads[1, i], grads[2, i]
            B[:, 3 * i : 3 * i + 3] = [
                [bi, 0, 0],
                [0, ci, 0],
                [0, 0, di],
                [ci, bi, 0],
                [0, di, ci],
                [di, 0, bi],
            ]

        return B, volume

    except np.linalg.LinAlgError as e:
        raise ValueError(f"Failed to compute B matrix: {str(e)}")


def compute_stresses(
    nodes: np.ndarray, elements: np.ndarray, u: np.ndarray, material: Dict
) -> np.ndarray:
    """
    Compute von Mises stresses for each element.

    Args:
        nodes: Node coordinates
        elements: Element connectivity
        u: Displacement vector
        material: Material properties

    Returns:
        von_mises: Array of von Mises stresses per element
    """
    # Material properties
    E = material["youngs_modulus_gpa"] * 1e9
    nu = material["poissons_ratio"]

    # Elastic constants
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # Constitutive matrix
    C = np.array(
        [
            [lambda_ + 2 * mu, lambda_, lambda_, 0, 0, 0],
            [lambda_, lambda_ + 2 * mu, lambda_, 0, 0, 0],
            [lambda_, lambda_, lambda_ + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ]
    )

    von_mises = np.zeros(len(elements))

    for i, el in enumerate(elements):
        try:
            # Get element displacements
            el_nodes = nodes[el]
            ue = np.zeros(12)  # 3 DOFs per node
            for j in range(4):
                ue[3 * j : 3 * j + 3] = u[3 * el[j] : 3 * el[j] + 3]

            # Compute B matrix
            B, _ = compute_B_matrix(el_nodes)

            # Compute strains and stresses
            strain = B @ ue
            stress = C @ strain

            # Compute von Mises stress
            s11, s22, s33, s12, s23, s31 = stress
            von_mises[i] = np.sqrt(
                0.5
                * (
                    (s11 - s22) ** 2
                    + (s22 - s33) ** 2
                    + (s33 - s11) ** 2
                    + 6 * (s12**2 + s23**2 + s31**2)
                )
            )

        except ValueError as e:
            logger.warning(f"Skipping stress calculation for element {i}: {str(e)}")
            von_mises[i] = 0.0
            continue

    return von_mises
