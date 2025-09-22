def test_can_import_vcaml():
    import vcaml

    assert hasattr(vcaml, '__package__')
