import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { FEATURES } from './feature-schema';
import { PredictionService, PredictionResponse } from './prediction.service';

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private api = inject(PredictionService);

  readonly fields = FEATURES;
  readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);
  readonly examples = [
    { label: 'Legitimate example', url: 'https://www.southbankmosaics.com' },
    { label: 'Phishing example', url: 'http://login-apple.com/verify' },
  ];

  values: Record<string, string> = Object.fromEntries(FEATURES.map(f => [f.name, '']));
  loading = signal(false);
  error = signal('');
  result = signal<PredictionResponse | null>(null);

  useExample(url: string) {
    this.values['URL'] = url;
    this.result.set(null);
    this.error.set('');
  }

  submit() {
    this.error.set('');
    this.result.set(null);

    if (this.deployedStatic) {
      this.error.set('Static GitHub Pages demo: run Flask and Angular locally for predictions.');
      return;
    }

    const payload: Record<string, string> = {};
    for (const field of this.fields) {
      const value = (this.values[field.name] ?? '').trim();
      if (field.required && !value) {
        this.error.set(`Enter a value for ${field.label}`);
        return;
      }
      if (field.maxLength && value.length > field.maxLength) {
        this.error.set(`${field.label} is longer than ${field.maxLength} characters`);
        return;
      }
      payload[field.name] = value;
    }

    this.loading.set(true);
    this.api.predict(payload).subscribe({
      next: r => {
        this.result.set(r);
        this.loading.set(false);
      },
      error: err => {
        this.error.set(
          `API error ${err.status || ''}: ${err.error?.error || err.message || 'Check the Flask terminal'}`
        );
        this.loading.set(false);
      },
    });
  }
}