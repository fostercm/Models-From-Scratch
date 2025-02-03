import unittest
from fastapi.testclient import TestClient
from main import app

class TestDataRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test client before running tests."""
        cls.client = TestClient(app)

    def test_upload_training_data(self):
        """Test uploading valid training data."""
        payload = {
            "input_data": "feature1,feature2\n1,2\n3,4",
            "output_data": "label\n0\n1",
            "data_type": "training"
        }
        response = self.client.post("/upload-data", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "training data uploaded successfully"})

    def test_upload_prediction_data(self):
        """Test uploading valid prediction data."""
        payload = {
            "input_data": "feature1,feature2\n5,6\n7,8",
            "data_type": "prediction"
        }
        response = self.client.post("/upload-data", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "prediction data uploaded successfully"})

    def test_missing_output_for_training(self):
        """Test that missing output data for training returns a 400 error."""
        payload = {
            "input_data": "feature1,feature2\n1,2\n3,4",
            "data_type": "training"
        }
        response = self.client.post("/upload-data", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Training data requires both input and output data.", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()