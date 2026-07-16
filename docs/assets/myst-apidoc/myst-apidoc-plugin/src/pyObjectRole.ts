import type { RoleData, RoleSpec } from 'myst-common';

function pyObjectRole(name: string, doc: string): RoleSpec {
  return {
    name,
    doc,
    body: {
      type: String,
      doc: 'Python object name',
      required: true,
    },
    run(data: RoleData) {
      const target = data.body as string;
      const value = target.startsWith('~') ? target.slice(1).split('.').at(-1) ?? target : target;
      return [{ type: 'inlineCode', value }];
    },
  };
}

export const pyClassRole = pyObjectRole('class', 'Python class reference role');
export const pyMethodRole = pyObjectRole('meth', 'Python method reference role');
export const pyModuleRole = pyObjectRole('mod', 'Python module reference role');
export const pyDataRole = pyObjectRole('data', 'Python data reference role');
