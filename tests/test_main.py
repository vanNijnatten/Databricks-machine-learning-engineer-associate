import pytest
from pytest_mock.plugin import MockerFixture

from db_ml_associate import main


@pytest.fixture
def integer() -> int:
    return 1


@pytest.mark.slow
def test_setup_fails(integer: int, mocker: MockerFixture) -> None:
    class TestObject:
        def method_to_test(self, integer: int) -> None:
            pass

    # Arrange
    mocker.patch.object(TestObject, "method_to_test")

    # Act
    obj = TestObject()
    obj.method_to_test(integer)

    # Assert
    obj.method_to_test.assert_called_once_with(integer)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]


@pytest.mark.parametrize("to_pass", [(True), (True)])
def test_passes(to_pass: bool) -> None:
    assert to_pass


@pytest.mark.parametrize("to_pass", [(False), (False)])
def test_expect_exception(to_pass: bool) -> None:
    with pytest.raises(AssertionError):
        assert to_pass


def test_package_exports_main() -> None:
    assert callable(main)
    assert main() is None
