from .partial_vae import PartialVAE


class TransformerEncoderPVAE(PartialVAE):
    # This class exists to make it easier to maintain and use separate configs for Transformer encoder PVAE.
    @classmethod
    def name(self):
        return "transformer_encoder_pvae"

    @classmethod
    def _create(cls, model_id, variables, save_dir, device, **model_config_dict):
        assert model_config_dict["set_encoder_type"] == "transformer"
        return super(TransformerEncoderPVAE, cls)._create(model_id, variables, save_dir, device, **model_config_dict)
