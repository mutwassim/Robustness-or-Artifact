import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResponse {
  prediction: 'Legitimate' | 'Phishing';
  is_phishing: boolean;
  phishing_probability: number;
  probabilities: number[]; // [Legitimate, Phishing]
  task: string;
  model_version?: string;
  disclaimer?: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private http = inject(HttpClient);

  predict(payload: Record<string, string>): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>('/api/predict', payload);
  }
}