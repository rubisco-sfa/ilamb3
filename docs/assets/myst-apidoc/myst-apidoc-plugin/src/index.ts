import type { MystPlugin } from 'myst-common';
import { apiDirective } from './apiDirective.js';
import { versionAddedDirective } from './versionAdded.js';
import { funcRole } from './funcRole.js';
import { pyClassRole, pyDataRole, pyMethodRole, pyModuleRole } from './pyObjectRole.js';

const plugin: MystPlugin = {
  name: 'Plugin to document APIs (currently using fleece output)',
  author: 'mystmd developers',
  license: 'MIT',
  transforms: [],
  directives: [apiDirective, versionAddedDirective],
  roles: [funcRole, pyClassRole, pyMethodRole, pyModuleRole, pyDataRole],
};

export default plugin;
