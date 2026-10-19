export type Feature = {
  name: string;
  label: string;
  kind: 'text' | 'number' | 'category';
  required: boolean;
  placeholder?: string;
  maxLength?: number;
  options?: string[];
  min?: number;
  max?: number;
};

// Every raw input field expected by the model bundle (bundle["features"] = ["URL"])
export const FEATURES: Feature[] = [
  {
    name: 'URL',
    label: 'Website URL',
    kind: 'text',
    required: true,
    placeholder: 'https://www.example.com',
    maxLength: 2048,
  },
];