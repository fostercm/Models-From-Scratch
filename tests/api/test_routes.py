import unittest
import logging
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch, MagicMock
import json
import io

client = TestClient(app)
logging.disable(logging.CRITICAL)


class TestRoutes(unittest.TestCase):
    def test_health_check(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "up"})

    def test_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "ML API is running!"})

    @patch("api.services.getModel")
    def test_train(self, mock_get_model):
        """Test the /train endpoint with mock model"""

        # Mock the model object and its methods
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.predict = MagicMock(return_value=[[0] for _ in range(10)])
        mock_model.cost = MagicMock(return_value=0.1)
        mock_model.get_params = MagicMock(return_value={"weights": [1, 2, 3]})
        mock_get_model.return_value = mock_model

        # Create mock CSV data
        csv_content = "feature1,feature2\n1,2\n3,4\n5,6"
        indicator_file = io.BytesIO(csv_content.encode())
        response_file = io.BytesIO("target\n0\n1\n0".encode())

        # Send the request
        response = client.post(
            "/train",
            files={
                "indicator": ("indicator.csv", indicator_file, "text/csv"),
                "response": ("response.csv", response_file, "text/csv"),
            },
            data={"model_name": "Linear Regression", "language": "Python"},
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("model", response_data)
        self.assertIn("loss", response_data)
        self.assertIn("params", response_data)

    @patch("api.services.getModel")
    def test_predict(self, mock_get_model):
        """Test the /predict endpoint with mock model"""

        # Mock the model object and its methods
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[[0] for _ in range(5)])
        mock_model.load_params = MagicMock()
        mock_get_model.return_value = mock_model

        # Create mock input data
        csv_content = "feature1,feature2\n1,2\n3,4\n5,6"
        indicator_file = io.BytesIO(csv_content.encode())

        # Mock model JSON file
        model_params = json.dumps({"params": {"beta": [1, 2, 3]}})
        model_file = io.BytesIO(model_params.encode())

        # Send the request
        response = client.post(
            "/predict",
            files={
                "indicator": ("indicator.csv", indicator_file, "text/csv"),
                "model_file": ("model.json", model_file, "application/json"),
            },
            data={"model_name": "Linear Regression", "language": "Python"},
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("predictions", response_data)
        self.assertIsInstance(response_data["predictions"], list)
