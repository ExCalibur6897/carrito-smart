from __future__ import annotations

import sqlite3

import pytest

from carrito_smart.cart import CartService
from carrito_smart.database import Database, InventoryError
from carrito_smart.models import CartItem


@pytest.fixture()
def database(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    return database


def test_initial_products_are_seeded(database):
    products = database.list_products()
    assert len(products) == 6
    assert all(product.stock > 0 for product in products)


def test_inventory_is_only_discounted_after_checkout(database):
    product = database.list_products()[0]
    cart = CartService(database)
    cart.add_product(product.id)
    cart.add_product(product.id)

    assert database.get_product(product.id).stock == product.stock

    receipt = cart.checkout()

    assert receipt.total_cents == product.price_cents * 2
    assert database.get_product(product.id).stock == product.stock - 2
    assert cart.items == []


def test_failed_sale_rolls_back_every_change(database):
    first, second = database.list_products()[:2]
    impossible_items = [
        CartItem(first, 1),
        CartItem(second, second.stock + 1),
    ]

    with pytest.raises(InventoryError):
        database.complete_sale(impossible_items)

    assert database.get_product(first.id).stock == first.stock
    with sqlite3.connect(database.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM sale_items").fetchone()[0] == 0


def test_cart_total_and_manual_exit(database):
    product = database.list_products()[0]
    cart = CartService(database)
    cart.add_product(product.id)
    cart.add_product(product.id)
    cart.remove_product(product.id)

    assert len(cart.items) == 1
    assert cart.items[0].quantity == 1
    assert cart.total_cents == product.price_cents
