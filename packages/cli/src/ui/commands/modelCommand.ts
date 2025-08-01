/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { CommandKind, OpenDialogActionReturn, SlashCommand } from './types.js';
import { MessageType } from '../types.js';

export const modelCommand: SlashCommand = {
  name: 'model',
  description: 'Switch between different models',
  kind: CommandKind.BUILT_IN,
  action: (context, args) => {
    // 如果有参数，直接切换到指定模型
    if (args.trim()) {
      const modelName = args.trim();
      const customModels =
        context.services.config?.getAvailableCustomModels() || [];
      const modelExists = customModels.some(
        (model) => model.name === modelName,
      );

      if (modelExists) {
        // 异步切换模型
        context.services.config?.switchToCustomModel(modelName);
        context.ui.addItem(
          {
            type: MessageType.INFO,
            text: `Switched to model: ${modelName}`,
          },
          Date.now(),
        );
      } else {
        context.ui.addItem(
          {
            type: MessageType.ERROR,
            text: `Model '${modelName}' not found. Use '/model' without arguments to see available models.`,
          },
          Date.now(),
        );
      }
      return; // 返回 void 而不是 { type: 'handled' }
    }

    // 否则打开对话框选择模型
    return {
      type: 'dialog',
      dialog: 'model',
    } as OpenDialogActionReturn;
  },
};
