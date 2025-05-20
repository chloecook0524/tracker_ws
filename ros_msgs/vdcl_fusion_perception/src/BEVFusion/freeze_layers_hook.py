from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class FreezeLayersHook(Hook):
    def __init__(self, layers_to_freeze):
        self.layers_to_freeze = layers_to_freeze

    def before_train(self, runner):
        model = runner.model
        for layer_name in self.layers_to_freeze:
            layer = self.get_layer(model, layer_name)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = False
                runner.logger.info(f"Layer {layer_name} frozen.")
                # print(f"Layer {layer_name} frozen.")
            else:
                runner.logger.info(f"Layer {layer_name} not found.")
                # print(f"Layer {layer_name} no frozen.")

        params = filter(lambda p: p.requires_grad, runner.model.parameters())
        runner.optim_wrapper.optimizer.param_groups[0]['params'] = list(params)  # 첫 번째 param_group 수정
        runner.logger.info("Updated optimizer param_groups to include only trainable parameters.")

    def get_layer(self, model, layer_name):
        if hasattr(model, 'module'):
            model = model.module
        components = layer_name.split('.')
        layer = model
        for comp in components:
            if hasattr(layer, comp):
                layer = getattr(layer, comp)
            else:
                return None
        return layer

    