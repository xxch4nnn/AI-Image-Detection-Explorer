import unittest
from unittest.mock import MagicMock, patch
import sys

# Mocking missing dependencies
mock_st = MagicMock()
# Mock streamlit decorators to be identity decorators
# This handles both @st.cache_resource and @st.cache_resource()
def identity_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda func: func

mock_st.cache_resource = identity_decorator
mock_st.cache_data = identity_decorator

mock_pd = MagicMock()
mock_px = MagicMock()
mock_go = MagicMock()
mock_datasets = MagicMock()
mock_pil = MagicMock()
mock_np = MagicMock()
mock_torch = MagicMock()
mock_torchvision = MagicMock()
mock_requests = MagicMock()

sys.modules["streamlit"] = mock_st
sys.modules["pandas"] = mock_pd
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = mock_px
sys.modules["plotly.graph_objects"] = mock_go
sys.modules["datasets"] = mock_datasets
sys.modules["PIL"] = mock_pil
sys.modules["numpy"] = mock_np
sys.modules["torch"] = mock_torch
sys.modules["torchvision"] = mock_torchvision
sys.modules["torchvision.models"] = MagicMock()
sys.modules["torchvision.transforms"] = MagicMock()
sys.modules["requests"] = mock_requests

# Now we can import app
import app

class TestApp(unittest.TestCase):
    def setUp(self):
        # Reset global mocks before each test
        mock_st.reset_mock()
        mock_pd.reset_mock()

    def test_prepare_metadata_empty(self):
        # Mocking pandas.DataFrame to return a MagicMock
        with patch('pandas.DataFrame') as mock_df_cls:
            result = app.prepare_metadata([])
            self.assertIsNotNone(result)
            mock_df_cls.assert_called_once_with([])

    def test_prepare_metadata_with_data(self):
        dataset = [
            {'prompt': 'a photo of a cat', 'seed': 123, 'step': 50, 'cfg': 7.5, 'sampler': 'euler', 'width': 512, 'height': 512}
        ]
        with patch('pandas.DataFrame') as mock_df_cls:
            result = app.prepare_metadata(dataset)
            self.assertIsNotNone(result)
            # Check if the metadata was prepared correctly before passing to DataFrame
            args, _ = mock_df_cls.call_args
            prepared_data = args[0]
            self.assertEqual(len(prepared_data), 1)
            self.assertEqual(prepared_data[0]['prompt'], 'a photo of a cat')
            self.assertEqual(prepared_data[0]['index'], 0)

    def test_prepare_metadata_truncation(self):
        long_prompt = 'a' * 200
        dataset = [{'prompt': long_prompt}]
        with patch('pandas.DataFrame') as mock_df_cls:
            app.prepare_metadata(dataset)
            args, _ = mock_df_cls.call_args
            prepared_data = args[0]
            self.assertEqual(len(prepared_data[0]['prompt']), 100)

    @patch('app.load_dataset')
    def test_load_image_dataset_success(self, mock_load):
        mock_load.return_value = "mock_dataset"
        result = app.load_image_dataset()
        self.assertEqual(result, "mock_dataset")
        mock_load.assert_called_once_with("poloclub/diffusiondb", "2m_random_1k", split="train")

    @patch('app.load_dataset')
    def test_load_image_dataset_failure(self, mock_load):
        mock_load.side_effect = Exception("Loading error")
        result = app.load_image_dataset()
        self.assertIsNone(result)
        app.st.error.assert_called()

    @patch('app.models.resnet18')
    def test_load_classification_model_success(self, mock_resnet):
        mock_model = MagicMock()
        mock_resnet.return_value = mock_model

        model, preprocess = app.load_classification_model()

        self.assertEqual(model, mock_model)
        self.assertIsNotNone(preprocess)
        mock_model.eval.assert_called_once()

    @patch('app.models.resnet18')
    def test_load_classification_model_failure(self, mock_resnet):
        mock_resnet.side_effect = Exception("Model error")
        model, preprocess = app.load_classification_model()
        self.assertIsNone(model)
        self.assertIsNone(preprocess)
        app.st.warning.assert_called()

if __name__ == '__main__':
    unittest.main()
