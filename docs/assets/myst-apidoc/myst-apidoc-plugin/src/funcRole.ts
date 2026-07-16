import type { RoleData, RoleSpec } from 'myst-common';

export const funcRole: RoleSpec = {
  name: 'func',
  doc: 'Small function cross-reference role',
  body: {
    type: String,
    doc: 'Cross-reference target',
    required: true,
  },
  run(data: RoleData) {
    return [
      {
        type: 'link',
        url: `#${data.body as string}`,
        children: [
          {
            type: 'text',
            value: data.body as string,
          },
        ],
      },
    ];
  },
};
