from .vae_mixed import VAEMixed


class TransformerEncoderVAEM(VAEMixed):
    # This class exists to make it easier to use separate configs for transformer VAEM.
    @classmethod
    def name(self):
        return "transformer_encoder_vaem"

    @classmethod
    def _create(cls, model_id, variables, save_dir, device, **model_config_dict):
        assert model_config_dict["dep_set_encoder_type"] == "transformer"
        return super(TransformerEncoderVAEM, cls)._create(model_id, variables, save_dir, device, **model_config_dict)
