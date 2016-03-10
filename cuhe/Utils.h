/*
The MIT License (MIT)

Copyright (c) 2016 Andrea Peruffo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
  Helpers to serialize deserialize and manage ZZ and ZZX objects
*/

#pragma once
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <sstream>
#include <string>
#include <vector>
NTL_CLIENT

namespace cuHE_Utils {

  class Picklable {
    string key;
    string values;
    ZZX poly;
    ZZ* coeffs;
    int coeffs_len;
    string separator = ",";

    public:
      Picklable(string, ZZ*, int);
      Picklable(string, ZZX);
      Picklable(string);
      Picklable(string, string);
      Picklable(const Picklable&);
      ~Picklable();

      void setSeparator(string);
      string getSeparator();

      ZZX getPoly();
      ZZ* getCoeffs();
      int getCoeffsLen();

      string getKey();
      string getValues();

      string pickle();

    private:
      void setValuesString();
      void toCoeffs();
      void fromString(string);
  };

  class PicklableMap {
    vector<Picklable*> picklables;
    string separator = "#";

    public:
      PicklableMap(vector<Picklable*>);
      PicklableMap(string);
      PicklableMap(string, string);
      PicklableMap(string, string, string);
      ~PicklableMap();

      void setSeparator(string);
      string getSeparator();

      vector<Picklable*> getPicklables();
      string toString();

      Picklable* get(string);
    private:
      void fromString(string, string);
  };

}
