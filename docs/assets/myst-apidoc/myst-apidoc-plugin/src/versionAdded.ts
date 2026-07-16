import type { DirectiveData, DirectiveSpec } from 'myst-common';

export const versionAddedDirective: DirectiveSpec = {
  name: 'versionadded',
  alias: ['versionchanged', 'deprecated'],
  doc: 'Small version added/changed/deprecated directive',
  arg: {
    type: String,
    doc: 'Version the feature was added',
    required: true,
  },
  run(data: DirectiveData) {
    let verb: string;
    switch (data.name) {
      case 'versionchanged':
        verb = 'Changed';
        break;
      case 'deprecated':
        verb = 'Deprecated';
        break;
      default:
        verb = 'Added';
    }
    return [
      {
        type: 'admonition',
        kind: 'note',
        children: [
          {
            type: 'paragraph',
            children: [
              {
                type: 'text',
                value: `${verb} in Version ${data.arg as string}`,
              },
            ],
          },
        ],
      },
    ];
  },
};
