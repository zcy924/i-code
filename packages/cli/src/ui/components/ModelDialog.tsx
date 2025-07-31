/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { Colors } from '../colors.js';
import { type LoadedSettings } from '../../config/settings.js';
import { type Config } from '@google/gemini-cli-core';

interface ModelDialogProps {
  config: Config;
  settings: LoadedSettings;
  onSelect: (modelName: string) => void;
  onExit: () => void;
}

export const ModelDialog: React.FC<ModelDialogProps> = ({
  config,
  settings,
  onSelect,
  onExit,
}) => {
  const customModels = config.getAvailableCustomModels();
  const [selectedModelIndex, setSelectedModelIndex] = useState(0);

  useInput((input, key) => {
    if (key.escape || (key.ctrl && input === 'c')) {
      onExit();
    } else if (key.upArrow) {
      setSelectedModelIndex((prev) => 
        prev > 0 ? prev - 1 : customModels.length - 1
      );
    } else if (key.downArrow) {
      setSelectedModelIndex((prev) => 
        prev < customModels.length - 1 ? prev + 1 : 0
      );
    } else if (key.return) {
      if (customModels.length > 0) {
        onSelect(customModels[selectedModelIndex].name);
      }
    }
  });

  // 如果没有自定义模型，显示提示信息
  if (customModels.length === 0) {
    return (
      <Box 
        borderStyle="round" 
        borderColor={Colors.AccentYellow}
        paddingX={1}
        flexDirection="column"
      >
        <Text color={Colors.AccentYellow}>
          No custom models configured.
        </Text>
        <Text>
          Please configure custom models in your settings file.
        </Text>
      </Box>
    );
  }

  return (
    <Box 
      borderStyle="round" 
      borderColor={Colors.AccentBlue}
      paddingX={1}
      flexDirection="column"
    >
      <Text bold color={Colors.AccentBlue}>
        Select a Model:
      </Text>
      <Box flexDirection="column" marginTop={1}>
        {customModels.map((model, index) => (
          <Box key={model.name}>
            <Text color={index === selectedModelIndex ? Colors.AccentBlue : undefined}>
              {index === selectedModelIndex ? '▶ ' : '  '}
              {model.name} ({model.model})
            </Text>
          </Box>
        ))}
      </Box>
      <Box marginTop={1}>
        <Text dimColor>
          ↑/↓ to navigate, ↵ to select, Esc to cancel
        </Text>
      </Box>
    </Box>
  );
};