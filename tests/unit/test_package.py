def test_package_metadata() -> None:
    import pupil_labs.surface_tracker as this_project

    assert hasattr(this_project, "__version__")
