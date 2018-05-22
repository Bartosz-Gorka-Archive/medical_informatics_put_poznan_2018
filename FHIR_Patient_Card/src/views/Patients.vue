<template>
  <main class="l-app__main" role="main">

    <header class="c-toolbar">
      <div class="c-cell">
        <div class="c-cell__media">
          <i class="icon-footprint"></i>
        </div>
        <div class="c-cell__content">
          <h1 class="c-toolbar__title">Patients list</h1>
          <ol class="c-breadcrumb">
            <li class="c-breadcrumb__item">Patients</li>
          </ol>
        </div>
      </div>
    </header>

    <div>
      <table class="table table--data">
        <thead>
          <tr>
            <th class="u-hiddenDown@md">No</th>
            <th>ID</th>
            <th>Family name</th>
            <th>Gender</th>
            <th>Birthdate</th>
            <th>Active</th>
          </tr>
        </thead>
        <tfoot>
          <tr>
            <th class="u-hiddenDown@md">No</th>
            <th>ID</th>
            <th>Family name</th>
            <th>Gender</th>
            <th>Birthdate</th>
            <th>Active</th>
          </tr>
        </tfoot>
        <tbody>
          <template v-for="(patient, index) in this.patients">
            <tr>
              <td data-label="No" class="u-hiddenDown@md">{{ index + 1 }}</td>
              <td data-label="ID">{{ patient.resource.id }}</td>
              <td data-label="Family name">{{ secureShowFamilyName(patient.resource) }}</td>
              <td data-label="Gender">{{ patient.resource.gender }}</td>
              <td data-label="Birthdate">{{ patient.resource.birthDate }}</td>
              <td data-label="Active">{{ patient.resource.active }}</td>
            </tr>
          </template>

        <infinite-loading
          v-if="loadingPatients"
          v-on:infinite="infiniteHandler"
          ref="infiniteLoading"
          spinner="bubbles">
        </infinite-loading>

        </tbody>
      </table>
    </div>

  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters, mapActions } = createNamespacedHelpers('patient')

  export default {
    name: 'PatientsView',
    computed: mapGetters(['loadingPatients', 'patients']),
    mounted () {
      mapActions(['clear'])
    },
    methods: {
      secureShowFamilyName (record) {
        if ('name' in record) {
          if (Array.isArray(record.name) && record.name.length >= 1) {
            if ('family' in record.name[0]) {
              return record.name[0].family
            }
          }
        }
        return '<details not found>'
      },
      infiniteHandler (state) {
        this.$store.dispatch('patient/getPatients')
        .then(data => {
          mapGetters(['loadingPatients', 'patients'])

          if (this.loadingPatients) {
            state.loaded()
          } else {
            state.complete()
          }
        })
      }
    }
  }
</script>
