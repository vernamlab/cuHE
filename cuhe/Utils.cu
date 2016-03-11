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

#include "Utils.h"

namespace cuHE_Utils {

  Picklable::Picklable(string k, ZZ* cs, int l) {
    key = k;
    coeffs = cs;
    coeffs_len = l;

    poly = *(new ZZX());
    for (int i=0; i < l; i++) {
      SetCoeff(poly, i, cs[i]);
    }
    setValuesString();
  }

  Picklable::Picklable(string k, ZZX p) {
    key = k;
    poly = p;

    toCoeffs();
    setValuesString();
  }

  Picklable::Picklable(string data) {
    fromString(data);
  }

  Picklable::Picklable(string data, string sep) {
    separator = sep;
    fromString(data);
  }

  Picklable::Picklable( const Picklable &obj) {
    separator = obj.separator;
    key = obj.key;
    poly = obj.poly;

    toCoeffs();
    setValuesString();
  }

  void Picklable::toCoeffs() {
    coeffs_len = deg(poly)+1;
    coeffs = new ZZ[coeffs_len];
    for (int i=0; i < coeffs_len; i++) {
      coeffs[i] = coeff(poly, i);
    }
  }

  void Picklable::fromString(string data) {
    char * pch;
    char *saveptr;
    pch = strtok_r ((char*)data.c_str(), (char*)separator.c_str(), &saveptr);
    int i = -1;

    poly = *(new ZZX());
    while (pch != NULL) {
      string str(pch);

      if (i == -1)
        key = str;
      else
        SetCoeff(poly, i, conv<ZZ>(str.c_str()));

      pch = strtok_r (NULL, (char*)separator.c_str(), &saveptr);
      i++;
    }

    toCoeffs();
    setValuesString();
  }

  void Picklable::setValuesString() {
    stringstream buffer;

    for (int i=0; i < coeffs_len; i++) {
      buffer << coeffs[i];

      if (i != (coeffs_len-1))
        buffer << separator;
    }

    values.assign(buffer.str());
  }

  Picklable::~Picklable() {
    delete [] coeffs;
    poly.~ZZX();
    delete &key;
    delete &values;
  }

  void Picklable::setSeparator(string s) {
    separator = s;
    setValuesString();
  }

  string Picklable::getSeparator() {
    return separator;
  }

  ZZX Picklable::getPoly() {
    return poly;
  }

  ZZ* Picklable::getCoeffs() {
    return coeffs;
  }

  int Picklable::getCoeffsLen(){
    return coeffs_len;
  }

  string Picklable::getKey() {
    return key;
  }
  
  string Picklable::getValues() {
    return values;
  }

  string Picklable::pickle() {
    stringstream buffer;

    buffer << key << separator << values;
    return buffer.str();
  }


  PicklableMap::PicklableMap(vector<Picklable*> ps) {
    picklables = ps;
  }

  void PicklableMap::fromString(string data, string psep) {
    char * pch;
    char *saveptr;
    pch = strtok_r ((char*)data.c_str(), (char*)separator.c_str(), &saveptr);

    picklables.clear();
    
    while (pch != NULL) {
      string str(pch);

      picklables.push_back(new Picklable(str, psep));

      pch = strtok_r (NULL, (char*)separator.c_str(), &saveptr);
    }
  }

  PicklableMap::PicklableMap(string data) {
    fromString(data, ",");
  }

  PicklableMap::PicklableMap(string data, string psep) {
    fromString(data, psep);
  }

  PicklableMap::PicklableMap(string data, string sep, string psep) {
    separator = sep;
    fromString(data, psep);
  }

  PicklableMap::~PicklableMap() {
    picklables.clear();
  }

  void PicklableMap::setSeparator(string sep) {
    separator = sep;
  }
    
  string PicklableMap::getSeparator() {
    return separator;
  }

  vector<Picklable*> PicklableMap::getPicklables() {
    return picklables;
  }

  string PicklableMap::toString() {
    stringstream buffer;

    for (uint i=0; i < picklables.size(); i++) {
      buffer << picklables.at(i)->pickle();
      
      if (i != (picklables.size()-1))
        buffer << separator;
    }
    return buffer.str();
  }

  Picklable* PicklableMap::get(string key) {
    for (uint i=0; i < picklables.size(); i++) {
      if (picklables.at(i)->getKey().compare(key) == 0) {
        return picklables.at(i);
      }
    }
    throw "not found";
  }

} //namespace cuHE_Utils
