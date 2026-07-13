import type { MystPlugin } from 'myst-common';
import { apiDirective } from './apiDirective.js';
import { versionAddedDirective } from './versionAdded.js';
import { funcRole } from './funcRole.js';

const plugin: MystPlugin = {
  name: 'Plugin to document APIs (currently using fleece output)',
  author: 'mystmd developers',
  license: 'MIT',
  transforms: [],
  directives: [apiDirective, versionAddedDirective],
  roles: [funcRole],
};

export default plugin;
