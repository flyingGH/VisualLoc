import sys
sys.path.append('/Users/olivergrainge/Documents/github/VisualLoc')

from PlaceRec.Datasets import SFU
import numpy as np
import pytest


datasets = [SFU]

@pytest.mark.parametrize("dataset", datasets)
def test_name(dataset):
    ds = dataset()
    assert isinstance(ds.name, str)
    assert ds.name.islower()


@pytest.mark.parametrize("dataset", datasets)
def test_query_load(dataset):
    ds = dataset()
    assert len(ds.query_images("train")) > 1
    assert len(ds.query_images("val")) > 1
    assert len(ds.query_images("test")) > 1
    assert len(ds.query_images("all")) > 1


@pytest.mark.parametrize("dataset", datasets)
def test_query_partition(dataset):
    ds = dataset()
    length = 0
    length += len(ds.query_images("train"))
    length += len(ds.query_images("val"))
    length += len(ds.query_images("test"))
    assert len(ds.query_images("all")) == length


@pytest.mark.parametrize("dataset", datasets)
def test_query_sequence(dataset):
    ds = dataset()
    Q0, Q1 = ds.query_images("train"), ds.query_images("train")
    assert (Q0 == Q1).all()
    assert Q0.dtype  == np.uint8

    Q0, Q1 = ds.query_images("val"), ds.query_images("val")
    assert (Q0 == Q1).all()

    Q0, Q1 = ds.query_images("test"), ds.query_images("test")
    assert (Q0 == Q1).all()

    Q0, Q1 = ds.query_images("all"), ds.query_images("all")
    assert (Q0 == Q1).all()


@pytest.mark.parametrize("dataset", datasets)
def test_map_sequence(dataset):
    ds = dataset()
    M0, M1 = ds.map_images(), ds.map_images()
    assert (M0 == M1).all()
    assert M0.dtype == np.uint8



@pytest.mark.parametrize("dataset", datasets)
def test_map_gt(dataset):
    ds = dataset()
    gt_hard = ds.ground_truth(partition="all", gt_type="hard")
    gt_soft = ds.ground_truth(partition="all", gt_type="soft")
    assert np.sum(gt_soft) > np.sum(gt_hard)
    assert gt_hard.dtype == bool
    assert gt_soft.dtype == bool


@pytest.mark.parametrize("dataset", datasets)
def test_map_size(dataset):
    ds = dataset()
    gt_hard = ds.ground_truth(partition="train", gt_type="hard")
    gt_soft = ds.ground_truth(partition="train", gt_type="soft")
    Qn = len(ds.query_images("train"))
    assert gt_hard.shape[1]== Qn
    assert gt_soft.shape[1]== Qn

    gt_hard = ds.ground_truth(partition="val", gt_type="hard")
    gt_soft = ds.ground_truth(partition="val", gt_type="soft")
    Qn = len(ds.query_images("val"))
    assert gt_hard.shape[1]== Qn
    assert gt_soft.shape[1]== Qn

    gt_hard = ds.ground_truth(partition="test", gt_type="hard")
    gt_soft = ds.ground_truth(partition="test", gt_type="soft")
    Qn = len(ds.query_images("test"))
    assert gt_hard.shape[1]== Qn
    assert gt_soft.shape[1]== Qn

    gt_hard = ds.ground_truth(partition="all", gt_type="hard")
    gt_soft = ds.ground_truth(partition="all", gt_type="soft")
    Qn = len(ds.query_images("all"))
    assert gt_hard.shape[1]== Qn
    assert gt_soft.shape[1]== Qn
    













