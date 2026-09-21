from pathlib import Path
import gzip
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deg_zinb import read_cellranger, prepare_zipert_inputs, fit_glm
from deg_zinb.torch_backend.fit import FitConfig


@pytest.fixture
def cellranger_files(tmp_path):
    pytest.importorskip("scanpy")
    counts = np.array([
        [9, 1, 5, 0, 0], [18, 2, 0, 6, 1],
        [7, 3, 8, 0, 0], [12, 0, 0, 9, 0],
        [5, 1, 7, 5, 0], [4, 2, 4, 0, 0],
    ])
    ids = ["ENSG1", "ENSG2", "t", "c", "other"]
    names = ["GENE", "MT-TEST", "target", "control", "other"]
    types = ["Gene Expression"] * 2 + ["CRISPR Guide Capture"] * 3
    barcodes = [f"cell{i}-1" for i in range(6)]
    matrix = sparse.csc_matrix(counts.T)
    h5 = tmp_path / "filtered_feature_bc_matrix.h5"
    with h5py.File(h5, "w") as handle:
        group = handle.create_group("matrix")
        for key, value in dict(data=matrix.data, indices=matrix.indices,
                               indptr=matrix.indptr, shape=matrix.shape,
                               barcodes=np.asarray(barcodes, dtype="S")).items():
            group.create_dataset(key, data=value)
        features = group.create_group("features")
        for key, value in dict(id=ids, name=names, feature_type=types,
                               genome=["GRCh38"] * 5).items():
            features.create_dataset(key, data=np.asarray(value, dtype="S"))
    mtx = tmp_path / "filtered_feature_bc_matrix"
    mtx.mkdir()
    with gzip.open(mtx / "matrix.mtx.gz", "wb") as stream:
        mmwrite(stream, matrix)
    with gzip.open(mtx / "features.tsv.gz", "wt") as stream:
        stream.write("".join(f"{i}\t{n}\t{t}\n" for i, n, t in zip(ids, names, types)))
    with gzip.open(mtx / "barcodes.tsv.gz", "wt") as stream:
        stream.write("\n".join(barcodes) + "\n")
    return h5, mtx, counts


def test_h5_mtx_and_outs_agree(cellranger_files):
    h5, mtx, counts = cellranger_files
    for path in (h5, mtx, h5.parent):
        data = read_cellranger(path)
        assert sparse.issparse(data.X)
        np.testing.assert_array_equal(data.X.toarray(), counts[:, :2])
        np.testing.assert_array_equal(data.obsm["guide_counts"].toarray(), counts[:, 2:])
        assert list(data.var_names) == ["ENSG1", "ENSG2"]
        assert data.obs.iloc[1].grna_n_nonzero == 2
        assert data.obs.iloc[0].percent_mt == 10


def test_assignment_alignment_and_fit(cellranger_files, tmp_path):
    h5, _, counts = cellranger_files
    data = read_cellranger(h5)
    subset, design = prepare_zipert_inputs(
        data, target_guides=["t"], control_guides=["c"], covariates=(),
    )
    assert subset.n_obs == 4
    assert list(subset.obs.guide_id) == ["t", "c", "t", "c"]
    assert design.index.equals(subset.obs_names)
    np.testing.assert_array_equal(subset.X.toarray(), counts[:4, :2])
    result = fit_glm(subset, ["ENSG1"], design,
                     fit_cfg=FitConfig(max_iter=10, n_runs=1))
    assert "ENSG1" in result
    subset.write_h5ad(tmp_path / "input.h5ad")
    design.to_csv(tmp_path / "design.csv", index_label="cell_id")
    assert pd.read_csv(tmp_path / "design.csv", index_col=0).index.equals(
        ad.read_h5ad(tmp_path / "input.h5ad").obs_names)


def test_multi_inlet_and_log_covariates(cellranger_files):
    h5, mtx, _ = cellranger_files
    data = read_cellranger({"a": h5, "b": mtx})
    subset, design = prepare_zipert_inputs(
        data, target_guides=["t"], control_guides=["c"],
        covariates=("grna_n_umis",),
    )
    assert subset.n_obs == 8 and subset.obs_names.is_unique
    assert list(design.columns) == ["Intercept", "group", "grna_n_umis", "inlet_b"]
    np.testing.assert_allclose(design.grna_n_umis, np.log1p(subset.obs.grna_n_umis))


def test_validation_and_qc(cellranger_files):
    data = read_cellranger(cellranger_files[0])
    kwargs = dict(target_guides=["t"], control_guides=["c"], covariates=())
    subset, _ = prepare_zipert_inputs(data, max_percent_mt=15, **kwargs)
    assert subset.n_obs == 3
    with pytest.raises(ValueError, match="Unknown guide"):
        prepare_zipert_inputs(data, target_guides=["absent"], control_guides=["c"])
    with pytest.raises(ValueError, match="rank deficient"):
        prepare_zipert_inputs(data, target_guides=["t"], control_guides=["c"])
    with pytest.raises(ValueError, match="Both target and control"):
        prepare_zipert_inputs(data, max_percent_mt=0, **kwargs)
    data.obs.loc[:, "inlet"] = ["a", "b", "a", "b", "a", "b"]
    with pytest.raises(ValueError, match="rank deficient"):
        prepare_zipert_inputs(data, **kwargs)


def test_reordered_features_align_by_id(cellranger_files):
    h5, mtx, counts = cellranger_files
    order = [4, 1, 3, 0, 2]
    reordered = sparse.csc_matrix(counts[:, order].T)
    with h5py.File(h5, "r+") as handle:
        matrix = handle["matrix"]
        for key in ("data", "indices", "indptr"):
            del matrix[key]
            matrix.create_dataset(key, data=getattr(reordered, key))
        for dataset in matrix["features"].values():
            dataset[:] = dataset[:][order]
    data = read_cellranger({"original": mtx, "reordered": h5})
    np.testing.assert_array_equal(data.X[:6].toarray(), data.X[6:].toarray())
    np.testing.assert_array_equal(data.obsm["guide_counts"][:6].toarray(),
                                  data.obsm["guide_counts"][6:].toarray())
    with h5py.File(h5, "r+") as handle:
        handle["matrix/features/id"][0] = b"DIFFERENT"
    with pytest.raises(ValueError, match="same gene and guide"):
        read_cellranger({"original": mtx, "changed": h5})


def test_invalid_counts_rejected(cellranger_files):
    data = read_cellranger(cellranger_files[0])
    data.obsm["guide_counts"].data[0] = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        prepare_zipert_inputs(data, target_guides=["t"], control_guides=["c"])
