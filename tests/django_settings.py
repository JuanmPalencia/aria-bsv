"""Minimal Django settings for integration tests."""
SECRET_KEY = "aria-test-secret-key-not-for-production"
INSTALLED_APPS = ["django.contrib.contenttypes", "django.contrib.auth"]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
